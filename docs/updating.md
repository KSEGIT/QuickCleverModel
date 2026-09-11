# Updating the stack

Linux and Docker only. On macOS, run `git pull` then `make restart`.

## What `update.sh` does

1. **Checks first.** It stops if a tool it needs is missing, if the weights
   are missing, if `.env` is missing, or if a stack is already running from a
   different folder.
2. **Tests the stack before changing it.** This proves the state it is about
   to save really works. It happens before every other decision on purpose:
   it is the only weekly check a machine that never changes ever gets.
3. **Saves a way back.** It writes the commit to `run/update-state` with
   `verified=1` if step 2 passed, and gives the current image a second name,
   `bonsai-llama:rollback`.
4. **Refuses a commit that already failed.** See "The bad commit pin" below.
   This comes after the test above, not before it, so a machine with a pinned
   commit still gets checked every week.
5. **Updates.** Fetch, move to the commit it checked in step 4 with
   `git merge --ff-only`, rebuild the llama image, pull Open WebUI, restart.
6. **Proves it works.** It checks the chat UI answers, then asks every model
   in `/v1/models` for a short answer. Both halves matter: the Open WebUI
   image is a moving tag, so it can change even when no model does.
7. **Undoes the update if that fails.** It puts back the old commit and the
   old images, restarts, and tests again. Then it exits with an error.

If any part of the undo fails, it says so. A stack that answers is not the
same as a stack that was put back: a bad commit can still be checked out.

## Why it tests every model

On 10 September 2026 this stack reported `Up 29 hours (healthy)`, and
`/health` returned `{"status":"ok"}`, while every model failed to load.

Someone moved the repository while the stack was running:

```
mv QuickCleverModel/ ../firesandbrain1/
```

The container still pointed at the old folder. Docker made that folder again,
empty. So `/app/models` held no weights. The containers were alive. They just
could not answer.

A health check cannot see this. A real answer can. That is why step 6 loads
every model, and why step 1 refuses to start when the weights folder is empty.

## Why it tests every alias, not just one

The router starts each alias as its own process with its own settings. One
model that answers tells you nothing about the next one.

## Reading the reply

Bonsai thinks before it answers. A short reply can run out of tokens while it
is still thinking. Then `content` is empty and the text sits in
`reasoning_content`. That is a good answer. `smoke_ok` counts both, so a
working model is never rolled back by mistake.

## The bad commit pin

Undoing an update only moves the branch back. It does not stop the next run
from pulling the same commit again.

Without a pin, one bad commit upstream takes the machine down every week:
pull, fail, undo, wait, pull the same commit, fail again.

So when the new code itself fails, `update.sh` writes that commit to
`run/update-state` as `bad=`. The next run stops if the newest commit
upstream is still that one. It starts working again on its own as soon as a
newer commit lands.

Two failures count as proof the commit is bad:

- a model cannot answer, or
- the stack never becomes healthy after a clean build and start.

Other failures do not pin. A build that lost the network, or a start that
lost a race with the second network address, says nothing about the commit.
A wrong pin would stop every later update until a person cleared it.

To try the same commit again anyway:

```bash
sed -i '/^bad=/d' run/update-state
```

## Tuning the deadlines

`update.sh` rolls the stack back when it decides a model is broken, and
remembers the commit when the code is to blame. So a deadline that is too
short on a slow machine blocks good code until a person clears it.

Four settings in `.env` control this. See `.env.example` for each one:

| Setting | Default | What it bounds |
|---|---|---|
| `BONSAI_SMOKE_TIMEOUT` | 900s | One model's answer |
| `BONSAI_SMOKE_SWEEP_MAX` | 4x the above | One whole sweep, retries included |
| `BONSAI_HEALTH_TIMEOUT` | 180s | Waiting for `/health` after a restart |
| `BONSAI_SMOKE_RETRY_SLEEP` | 15s | Backoff when the router says it is busy |

If you raise the first two, raise `TimeoutStartSec` in
`docker/bonsai-update.service` as well. Its comment shows the arithmetic.

## The weekly timer

`docker/bonsai-update.timer` runs the update once a week.

- It starts Monday at 04:00 and waits a random time of up to 4 hours, so many
  machines do not hit the registry at the same second.
- `Persistent=true` means a machine that was off still updates when it
  starts again.
- It runs the update script at low priority. Note this does not slow the
  image build itself: Docker builds inside its own daemon, which the timer
  cannot reach.

Install both files as shown in the README. Check it with
`systemctl list-timers bonsai-update.timer`.

## When it will not update

`update.sh` fetches, then moves with `git merge --ff-only` to the exact
commit it checked against the pin. It does not use `git pull`, because `git
pull` runs its own fetch and can land on a newer commit than the one just
checked — including the pinned bad one.

If someone changed a file that the new commit also changes, the merge stops
and nothing else happens. Fix the checkout by hand.

Changes to other files do **not** stop it: `git merge --ff-only` only refuses
when the paths clash. Those changes survive the merge, and if the script then
has to undo the move it puts them in a stash first. Get them back with:

```bash
git -C /path/to/QuickCleverModel stash list
git -C /path/to/QuickCleverModel stash pop
```

The script says so in its log when it does this. Nothing is deleted.
This is on purpose: a machine that quietly throws away local changes is worse
than one that stops and asks.
