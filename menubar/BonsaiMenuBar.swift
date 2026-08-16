// Menu bar control for the Bonsai stack.
//
// A thin client over ./stack.sh — it holds no knowledge of ports, PIDs or
// start order. All state comes from `./stack.sh status --json`; the
// human-readable table is deliberately not parsed.
//
// Built by `make menubar`, which bakes the repo path into Info.plist as
// BonsaiRoot so the app can find stack.sh regardless of where the bundle sits.
import SwiftUI

// MARK: - Model

struct ServiceStatus: Decodable, Identifiable {
    let service: String
    let port: Int
    let state: String
    let pid: Int?

    var id: String { service }
    var isUp: Bool { state == "up" }
}

enum StackState {
    case allUp, allDown, partial, unknown
}

final class StackModel: ObservableObject {
    @Published private(set) var services: [ServiceStatus] = []
    @Published private(set) var scriptMissing = false
    /// True once the first `status --json` round-trip has completed (or been
    /// short-circuited by a missing script), so the brief window before the
    /// very first refresh lands doesn't masquerade as a real warning.
    @Published private(set) var hasLoaded = false
    /// Non-nil while an action is running, e.g. "Restarting…".
    @Published private(set) var busy: String?
    /// Non-nil after an action exits non-zero. Cleared when the next starts.
    @Published private(set) var failure: String?

    let root: String
    private var timer: Timer?

    init(root: String) {
        self.root = root
        refresh()
        // 15s, not 1s: each status shells out to lsof once per service.
        // Scheduled manually (not via .scheduledTimer) and added in .common
        // modes: while the MenuBarExtra dropdown is open, AppKit runs the
        // run loop in .eventTracking mode, and .default-mode timers (what
        // .scheduledTimer uses) simply don't fire during that window.
        let t = Timer(timeInterval: 15, repeats: true) { [weak self] _ in
            self?.refresh()
        }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    var scriptPath: String { root + "/stack.sh" }

    var state: StackState {
        if scriptMissing || services.isEmpty { return .unknown }
        let up = services.filter { $0.isUp }.count
        if up == services.count { return .allUp }
        if up == 0 { return .allDown }
        return .partial
    }

    var symbolName: String {
        switch state {
        case .allUp:   return "leaf.fill"
        case .allDown: return "leaf"
        case .partial: return "exclamationmark.triangle"
        // Warn for a real problem (missing script, or a completed refresh
        // that still came back empty/undecodable) — not for the brief
        // window before the very first refresh has landed.
        case .unknown: return (scriptMissing || hasLoaded) ? "exclamationmark.triangle"
                                                            : "leaf"
        }
    }

    var summary: String {
        if let busy { return busy }
        if let failure { return failure }
        switch state {
        case .allUp:   return "Bonsai stack — running"
        case .allDown: return "Bonsai stack — stopped"
        case .partial: return "Bonsai stack — partial"
        case .unknown: return scriptMissing ? "⚠ stack.sh not found"
                                            : "Bonsai stack — unknown"
        }
    }

    func refresh() {
        let path = scriptPath
        guard FileManager.default.isExecutableFile(atPath: path) else {
            scriptMissing = true
            services = []
            hasLoaded = true
            return
        }
        scriptMissing = false
        DispatchQueue.global(qos: .utility).async {
            let (_, out) = StackModel.capture(path, ["status", "--json"])
            let parsed = (try? JSONDecoder().decode([ServiceStatus].self,
                                                    from: Data(out.utf8))) ?? []
            DispatchQueue.main.async {
                self.services = parsed
                self.hasLoaded = true
            }
        }
    }

    /// Runs a stack.sh verb off the main thread. Never blocks the UI.
    func perform(_ verb: String, label: String) {
        guard busy == nil else { return }   // one action at a time
        busy = label
        failure = nil
        let path = scriptPath
        DispatchQueue.global(qos: .userInitiated).async {
            let (code, _) = StackModel.capture(path, [verb])
            DispatchQueue.main.async {
                self.busy = nil
                if code != 0 {
                    self.failure = "⚠ \(label) failed — see logs"
                }
                self.refresh()
            }
        }
    }

    func openLogs() {
        NSWorkspace.shared.selectFile(nil,
            inFileViewerRootedAtPath: root + "/run/logs")
    }

    /// Runs `path args` to completion. Returns (exit code, stdout).
    ///
    /// Bounded by `timeout`: a wedged child (e.g. a hung stack.sh) must never
    /// block the caller forever — `perform()`'s completion block, and with it
    /// `busy`, must always eventually run so the action buttons re-enable.
    /// 600s sits comfortably above stack.sh's own ~240s readiness loop and
    /// the minutes a cold `up` can legitimately take mapping the model, so a
    /// normal slow restart never trips it.
    static func capture(_ path: String, _ args: [String], timeout: TimeInterval = 600) -> (Int32, String) {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: path)
        // The app is launched by LaunchServices with cwd "/", not the repo
        // root. stack.sh itself resolves paths via dirname($0) so that's
        // fine, but children it execs (e.g. open-webui, whose secret-key
        // file defaults to Path.cwd()/.webui_secret_key) inherit our cwd
        // verbatim. Pin it to the script's directory so `up`/`restart`
        // behave the same launched from the app as from a repo-root shell.
        proc.currentDirectoryURL = URL(fileURLWithPath: path).deletingLastPathComponent()
        proc.arguments = args
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        do { try proc.run() } catch { return (-1, "") }

        // Force-terminate if `proc` is still running once `timeout` elapses.
        // Killing it closes its inherited pipe fds, which is what unblocks
        // the read below if the child has wedged without exiting.
        let watchdog = DispatchWorkItem {
            if proc.isRunning { proc.terminate() }
        }
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + timeout, execute: watchdog)

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        watchdog.cancel()

        // A process we killed via the watchdog exits via uncaught SIGTERM,
        // not a normal exit — treat that (and any other signal death) as
        // failure regardless of the raw status number, so perform() always
        // takes its failure path and resets `busy` rather than trusting a
        // signal number that happens to be zero mod 256.
        if proc.terminationReason == .uncaughtSignal {
            return (-1, String(data: data, encoding: .utf8) ?? "")
        }
        return (proc.terminationStatus, String(data: data, encoding: .utf8) ?? "")
    }
}

// MARK: - Menu

struct MenuContent: View {
    @ObservedObject var model: StackModel

    var body: some View {
        Text(model.summary)
        Divider()
        ForEach(model.services) { svc in
            Text("\(svc.isUp ? "●" : "○")  \(svc.service)  :\(String(svc.port))")
        }
        Divider()
        Button("Open chat UI") { model.perform("open", label: "Opening…") }
            .disabled(model.busy != nil || model.scriptMissing)
        Button("Restart") { model.perform("restart", label: "Restarting…") }
            .disabled(model.busy != nil || model.scriptMissing)
        Button("Stop") { model.perform("down", label: "Stopping…") }
            .disabled(model.busy != nil || model.scriptMissing)
        if model.failure != nil {
            Button("Open logs") { model.openLogs() }
        }
        Divider()
        Button("Quit") { NSApplication.shared.terminate(nil) }
            .keyboardShortcut("q")
    }
}

// MARK: - App

@main
struct BonsaiMenuBarApp: App {
    @StateObject private var model = StackModel(root: BonsaiMenuBarApp.repoRoot())

    static func repoRoot() -> String {
        Bundle.main.object(forInfoDictionaryKey: "BonsaiRoot") as? String ?? ""
    }

    var body: some Scene {
        MenuBarExtra {
            MenuContent(model: model)
        } label: {
            Image(systemName: model.symbolName)
        }
        .menuBarExtraStyle(.menu)
    }
}
