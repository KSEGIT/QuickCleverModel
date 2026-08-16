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

    let root: String
    private var timer: Timer?

    init(root: String) {
        self.root = root
        refresh()
        // 15s, not 1s: each status shells out to lsof once per service.
        timer = Timer.scheduledTimer(withTimeInterval: 15, repeats: true) { [weak self] _ in
            self?.refresh()
        }
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
        default:       return "exclamationmark.triangle"
        }
    }

    var summary: String {
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
            return
        }
        scriptMissing = false
        DispatchQueue.global(qos: .utility).async {
            let (_, out) = StackModel.capture(path, ["status", "--json"])
            let parsed = (try? JSONDecoder().decode([ServiceStatus].self,
                                                    from: Data(out.utf8))) ?? []
            DispatchQueue.main.async { self.services = parsed }
        }
    }

    /// Runs `path args` to completion. Returns (exit code, stdout).
    static func capture(_ path: String, _ args: [String]) -> (Int32, String) {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: path)
        proc.arguments = args
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        do { try proc.run() } catch { return (-1, "") }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
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
