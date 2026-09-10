# Updating the stack

Linux and Docker only. On macOS, run `git pull` then `make restart`.

## What `update.sh` does

1. **Checks first.** It stops if the weights are missing, if `.env` is
   missing, or if a stack is already running from a different folder.
2. **Saves a way back.** It writes the current commit to `run/update-state`
   and gives the current image a second name, `bonsai-llama:rollback`.
3. **Updates.** `git pull --ff-only`, rebuild the llama image, pull
   Open WebUI, restart.
4. **Proves it works.** It asks every model in `/v1/models` for a short
   answer.
5. **Undoes the update if that fails.** It puts back the old commit and the
   old images, restarts, and tests again. Then it exits with an error.

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

A health check cannot see this. A real answer can. That is why step 4 loads
every model, and why step 1 refuses to start when the weights folder is empty.

## Why it tests every alias, not just one

The router starts each alias as its own process with its own settings. One
model that answers tells you nothing about the next one.

## Reading the reply

Bonsai thinks before it answers. A short reply can run out of tokens while it
is still thinking. Then `content` is empty and the text sits in
`reasoning_content`. That is a good answer. `smoke_ok` counts both, so a
working model is never rolled back by mistake.

## The weekly timer

`docker/bonsai-update.timer` runs the update once a week.

- It waits a random time of up to 4 hours, so many machines do not hit the
  registry at the same second.
- `Persistent=true` means a machine that was off still updates when it
  starts again.
- It runs at low priority, so it does not fight the model server for the CPU.

Install both files as shown in the README. Check it with
`systemctl list-timers bonsai-update.timer`.

## When it will not update

`update.sh` uses `git pull --ff-only`. If someone changed files on the
machine, the pull stops and nothing else happens. Fix the checkout by hand.
This is on purpose: a machine that quietly throws away local changes is worse
than one that stops and asks.
