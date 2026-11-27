# capy-flash app (alpha)

> **Alpha.** The app is new and has been used on few machines. Its settings, file layout and behaviour can change from
> one release to the next. It does not change how the engine generates: it starts the release's own launchers. If
> something goes wrong, the command line in the [README](../README.md#quick-start) does the same job; please
> [open an issue](https://github.com/ValerioDolci/capy-flash/issues) with what the app showed and its engine log.
> Not there yet: access from other machines on the network, unloading the model when idle, starting at login, and a
> copy of the app in the release tarball (get it from this repository).

A local window for capy-flash: it installs the engine and a model that fits your machine, starts and stops it, tunes
it to your hardware, and lets you chat with it — streaming, visible reasoning, the speed of every answer. It also
talks to any OpenAI-compatible server, which is how it works on a Mac. macOS and Linux. Python 3.9+ standard library
only: nothing to install.

## Start

The app is the `app/` folder of this repository: `git clone https://github.com/ValerioDolci/capy-flash` (or the
release's *Source code* zip), then, from the repository's folder:

```sh
app/launch/capy-app             # opens the window (a Chromium "app" window if Chrome/Chromium/Edge/Brave is installed)
app/launch/capy-app --install   # adds an icon: ~/Applications/capy-flash.app (macOS) or a menu/desktop entry (Linux)
```

`--install` copies the app to its own place (inside `~/Applications/capy-flash.app` on macOS, `~/.local/share/capy-flash/app`
on Linux), so the icon keeps working wherever this folder moves; run it again after an update. Starting the app again
while it runs just opens the window. *Settings → Quit the app* stops it, and the engine it started.

## First run

The app looks at the computer and proposes one thing:

- **A server already answers** (capy-flash on :8080, vramble on :8099, vLLM, LM Studio, Ollama): it uses that server
  as it is, and never starts an engine next to it.
- **Linux x86_64 with RTX 50-series GPUs of 16 GB and nothing answering**: *Install here* → pick a model (the one the
  README's "Pick a model" table gives for your GPUs, RAM, CPU and disk is preselected; the others say why they do not
  fit) → download and verify → the engine starts → optionally *Tune now*.
- **A Mac, or a machine without suitable GPUs**: the engine cannot run there (it needs Linux and NVIDIA `sm_120` GPUs
  with 16 GB), so the app asks for a server's address — for a server on another machine,
  `ssh -N -L 8080:127.0.0.1:8080 user@machine` and `http://127.0.0.1:8080/v1`.

Everything can be changed later from the status card at the bottom left.

## What it does

- **Chat**: streaming, reasoning (collapsible), thinking on/off, system prompt, images for models that read them
  (the app does not send images to Flash-Next, which is text only), conversations on disk (rename with a double
  click), copy / answer again. Under each answer: time to first token, tokens per second (the engine's own number
  when it reports `timings`), the token flow.
- **Engine** (Linux): start, stop, state and log of the engine the app installed. It is started with the release's
  launcher exactly as `install.sh` prints it, on `127.0.0.1`, restarted when it exits with status 2, and stopped when
  the app quits. The app refuses to start it when something already answers on its port, when another program holds
  the GPUs, or when the machine fails `install.sh`'s checks.
- **Models**: the capy-flash models, with what each needs. Downloads come from the project's GitHub release and
  Hugging Face repositories only, over HTTPS, resume where they stopped, and every file is checked against its
  published SHA-256 before anything is used. The layout is `install.sh`'s (`~/capy-flash/current`,
  `~/capy-flash/models/<name>/`): what the command line installed is recognised, and the other way round.
  Inkling 2 Mini (two 16 GB GPUs, text only) downloads like the others (20.3 GB, with its two drafters). It needs a
  release that ships `capy-serve-inkling2-mini.sh`, which turns the DSpark drafter on.
- **Files that are already there are yours.** The app deletes or replaces only files it created itself (it keeps a
  record of each, with size, time and digest, in `~/capy-flash/.capy-app/`, where its temporary files live too). A
  model folder it finds is hashed first, as one unit: if every file matches the repository it is used as it is (and
  *Delete* is not offered for it); if any file differs, nothing in it is adopted or changed — the app says that a
  different version is there and offers to download the current one into a new folder next to it
  (`models/<name>-<revision>`), or to use the folder as it is, unverified, if you say so. A folder that is a link
  leading outside `~/capy-flash` is never written to, deleted from or read for verification: it can only be used as
  it is. The same holds for releases: one the app did not install and that fails its checksums is left alone.
- **Tuning**: a short fixed benchmark (generation, prompt reading, first token of a short prompt; optionally a long
  context) run on the settings that keep the precision of the weights and of every operation (they can change the
  bits of an answer only through the order of the sums or where they run) — CPU threads for the experts,
  `--miss-split`, `--stream-min-tokens`, one or two GPUs, `--kv-ram-active` — each on its own, after a discarded warm-up request.
  The defaults are measured at the start and again at the end: the better of the two is the reference and their
  difference is the noise of the machine. A setting is kept only if it beats the reference by more than 3 % and by
  more than that noise, without making anything else slower; the kept ones are then measured together, and if together
  they do not hold up the single best one is used. The result
  is this machine's profile, used at every start; *Revert to the defaults* undoes it. The app says how long it takes before it starts and can be cancelled. The speed switches that change the
  numbers (`NINFER_TP_AR_PAYLOAD`, `NINFER_Q8_A8`, `NINFER_ATTN_PV_F16ACC`) are never set, and are removed from the
  engine's environment.
- **Tools → OpenCode**: *Open in OpenCode* opens a terminal in the chosen folder with `OPENCODE_CONFIG` pointing at
  the app's own `opencode.json` (a `capy-flash` provider with the chat models). Your global OpenCode config is merged,
  never edited. OpenCode 1.x format.
- **Tools → other tools**: ready-to-paste settings for Claude Code, the OpenAI Python SDK and curl.

English and Italian (the browser's language, or *Settings*); dark, light or the system's theme.

## Where things are

| | macOS | Linux |
|---|---|---|
| settings (`app.json`, 0600), tuning profile (`tuning.json`) | `~/Library/Application Support/capy-flash/` | `~/.config/capy-flash/` |
| conversations, engine log | same folder | `~/.local/share/capy-flash/` |
| engine and models | — | `~/capy-flash/` (`capy-app --engine-dir DIR` changes it, once); the app's records and temporary files in `~/capy-flash/.capy-app/` |

## Security

The app listens on 127.0.0.1 only. Every launch creates a session token: the launcher's URL sets it as an HttpOnly,
SameSite=Strict cookie, and the API refuses requests without it, with a foreign `Host` (DNS rebinding) or from
another origin. No CORS. Everything that installs, starts, stops or changes something is a POST, so it is checked for
its origin too. The page cannot choose what is downloaded, from where, or into which folder: sources and models are
fixed in `capy_app/catalog.py`, and the install folder is set from the command line only, because it decides which
launcher is executed. A downloaded file that fails its checksum is deleted and nothing is installed from it; a file
the app did not create is never deleted or overwritten, and nothing is written through a link out of the engine's
folder. The
endpoint's API key stays in the app process (the page never sees it) and OpenCode reads it from a 0600 file. Model
output is rendered by an escaping Markdown renderer under a strict Content Security Policy.

## How it works

```
capy_app/      Python standard library only
  server.py      the page, the JSON API (/api/*) and a streaming proxy to the backend's /v1
  backends.py    any OpenAI-compatible server; the engine on this computer is reached the same way
  local.py       ties the parts below to the settings, and decides who may do what
  machine.py     the probe (nvidia-smi, /proc) and install.sh's checks
  catalog.py     the release and the models the app can install, with their launchers and requirements
  downloads.py   HTTPS, resumable, checked against the published SHA-256
  installer.py   install.sh's layout, and the rules about files the app did not create
  engine.py      starts, watches and stops the engine
  tuning.py      the benchmark and the tuning run
  store.py, opencode.py, security.py, paths.py
web/           ES modules, no build step: util, i18n, md, state, chat, panel, setup, app
```

The engine goes `stopped → starting` (until `/health` answers) `→ running → stopping → stopped`. An exit with status
2 while it runs starts it again, at most 3 times in 10 minutes; any other exit is `failed`, with the last lines of its
log. If the app dies, the engine it started is found again at the next start (pid and command line) and adopted.

## Tests

```sh
cd app && python3 -m unittest discover -s tests
```

They run anywhere, without a GPU: the hardware probe reads fixture outputs of `nvidia-smi` and `/proc`, downloads go
to a local stand-in for GitHub and Hugging Face (`tests/fake_sources.py`), and the engine is `tests/fake_engine.py`,
started through a fake release with the real launchers' shape.
