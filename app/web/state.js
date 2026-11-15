// capy-flash app — what the page knows (settings, the server's models, the engine on this computer) and the one
// answer every view needs: where do the answers come from, and what state is that in?
import { api, bytes, duration } from "/util.js";
import { t, units, has } from "/i18n.js";

export const S = {
  version: "", settings: null, platform: { os: "", arch: "", can_run_engine: false }, opencode: null, home: "",
  models: [], modelsError: "", modelsLoaded: false,   // the chat models of the active backend
  eng: null,                                          // GET /api/engine (local mode, or while the panel shows it)
  conv: null, list: [], busy: null, attachments: [], folder: "",
};

const listeners = new Set();
export const onChange = (fn) => { listeners.add(fn); };
export const changed = () => { for (const fn of listeners) fn(); };

export const isLocal = () => !!S.settings && S.settings.backend.kind === "local";
export const isMac = () => S.platform.os === "Darwin";
export const chatModels = () => S.models.filter((m) => m.kind === "chat");
/** Does this model read images? true / false, or null when the app cannot tell (it then lets the server decide). */
export const visionOf = (id) => { const m = S.models.find((x) => x.id === id); return m ? m.vision : null; };

/** "Flash-Next" / "27B" plus the edition, for a catalog model. */
export function modelName(m, withEdition = true) {
  if (!m) return "";
  const base = m.title.replace(/^Qwen3\.8-/, "");
  return withEdition ? `${base} ${m.edition}` : base;
}
export const chosenModel = () => (S.eng && S.eng.models.find((m) => m.key === S.eng.model)) || null;

export async function loadState() {
  const state = await api("GET", "/api/state");
  S.version = state.version; S.settings = state.settings; S.platform = state.platform;
  S.opencode = state.opencode; S.home = state.home;
  return state;
}

export async function loadModels() {
  try {
    const answer = await api("GET", "/api/models");
    S.models = answer.models;
    S.modelsError = answer.error || "";
  } catch (e) {
    S.models = [];
    S.modelsError = e.message || "unreachable";
  }
  S.modelsLoaded = true;
  changed();
}

let engineTimer = 0;
let lastProcessState = "";
let watchers = 0;   // views that want the engine state fresh (the panel, the setup)

export async function loadEngine() {
  try { S.eng = await api("GET", "/api/engine"); } catch (_) { return S.eng; }
  const state = S.eng.process.state;
  if (state !== lastProcessState) {
    const was = lastProcessState;
    lastProcessState = state;
    if (isLocal() && (state === "running" || was === "running")) await loadModels();
  }
  changed();
  return S.eng;
}

export function engineBusy() {
  const e = S.eng;
  if (!e) return false;
  return (e.job && e.job.state === "running") || e.tuning.state === "running" ||
    ["starting", "stopping"].includes(e.process.state);
}

/** Keep the engine state fresh: every second while something moves, slowly otherwise; never in endpoint mode
 *  unless a view asked for it. */
export function scheduleEngine() {
  clearTimeout(engineTimer);
  if (!isLocal() && !watchers) return;
  engineTimer = setTimeout(async () => { await loadEngine(); scheduleEngine(); }, engineBusy() ? 1000 : watchers ? 3000 : 6000);
}
export function watchEngine(on) {
  watchers = Math.max(0, watchers + (on ? 1 : -1));
  scheduleEngine();
}

export async function act(path, body) {
  S.eng = await api("POST", path, body || {});
  lastProcessState = S.eng.process ? S.eng.process.state : lastProcessState;
  changed();
  scheduleEngine();
  return S.eng;
}

export function jobPercent(job) {
  if (!job) return 0;
  if (job.percent != null) return Math.floor(job.percent * 100);  // downloads, or the check of what is on disk
  return job.bytes_total ? Math.min(100, Math.floor((job.bytes_done / job.bytes_total) * 100)) : 0;
}

/** One line of install.sh's checks (GPU, driver, CPU, RAM, disk), in the user's language. */
export function checkText(line) {
  const p = Object.assign({}, line.params || {});
  let key = `chk_${line.code}_${line.level}`;
  if (line.code === "simd" && line.level === "ok") { key = "chk_simd_ok"; p.kind = p.kind === "avx512" ? "AVX-512" : "AVX2"; }
  if (line.code === "gpu" && line.level === "fail") p.found = (p.found || []).join(", ") || t("chk_gpu_none");
  if (line.code === "driver") p.version = p.version || "?";
  return has(key) ? t(key, p) : line.text;
}

/** "llama-server" once, however many processes of it hold the GPUs. */
export const busyNames = (processes) => [...new Set(processes.map((p) => p.name.split("/").pop()))].join(", ");

/** A known refusal or failure, in the user's language; the server's own sentence otherwise. */
export function errorText(error) {
  if (!error) return "";
  if (error.detail && error.detail.line) return checkText(error.detail.line);  // the machine check that failed, with its numbers
  const key = (["network", "disk", "checksum", "http", "layout", "insecure", "internal", "different_version", "outside",
    "foreign_release", "foreign_current"].includes(error.code) ? "err_" : "e_") + error.code;
  const known = t(key);
  if (known === key) return error.message || "";
  const busy = error.code === "gpu_busy" && error.detail && error.detail.processes;
  if (busy && busy.length) {  // name the program that holds the GPUs
    return known.replace(/\.\s/, ` (${busyNames(busy)}). `);
  }
  return known;
}

/** Where the answers come from and in what state: for the status card, the empty chat and the composer. */
export function status() {
  const u = units();
  const b = S.settings.backend;
  if (!isLocal()) {
    const server = b.name || b.base_url;
    const url = b.base_url.replace(/^https?:\/\//, "").replace(/\/v1$/, "");
    if (!S.modelsLoaded) return { kind: "checking", tone: "idle", card: [server, t("conn_checking")], title: "", sub: "", can: false };
    if (S.modelsError) {
      return { kind: "unreachable", tone: "bad", card: [server, t("card_unreachable")],
        title: t("st_unreachable_title", { server }), sub: t(isMac() ? "st_unreachable_mac" : "st_unreachable_sub", { url }),
        actions: ["retry", "connection"], can: false };
    }
    if (!chatModels().length) {
      return { kind: "nochat", tone: "warn", card: [server, t("no_models")], title: t("st_nochat_title", { server }),
        sub: t("st_nochat_sub"), actions: ["retry", "connection"], can: false };
    }
    return { kind: "ready", tone: "ok", card: [server, t("card_models", { n: chatModels().length })],
      title: t("st_ready_title"), sub: (model) => t("st_ready_remote", { model, server }), can: true };
  }
  const e = S.eng;
  if (!e) return { kind: "checking", tone: "idle", card: ["capy-flash", ""], title: "", sub: "", can: false };
  const m = chosenModel();
  const name = modelName(m, false) || "capy-flash";
  const full = modelName(m);
  const p = e.process, job = e.job, tune = e.tuning;
  const jobRunning = job && job.state === "running";
  if (tune.state === "running") {
    return { kind: "tuning", tone: "busy", card: [name, t("card_tuning", { n: Math.min(tune.done + 1, tune.total), m: tune.total })],
      title: t("st_tuning_title"), sub: t("st_tuning_sub", { n: Math.min(tune.done + 1, tune.total), m: tune.total,
        left: duration(tune.eta || 0, u) }), progress: tune.total ? tune.done / tune.total : 0, actions: ["tuning"], can: false };
  }
  if (p.state === "running" && chatModels().length) {
    const sub = jobRunning ? t("card_download", { pct: jobPercent(job) }) : t("card_running") + (m && m.tuned ? " · " + t("card_tuned") : "");
    return { kind: "ready", tone: "ok", card: [name, sub], title: t("st_ready_title"),
      sub: () => t("st_ready_local", { model: full }), can: true };
  }
  if (p.state === "starting" || p.state === "running") {
    const since = p.started_at ? Date.now() / 1000 - p.started_at : 0;
    return { kind: "starting", tone: "busy", card: [name, t("card_starting")], title: t("st_starting_title", { model: full }),
      sub: t("st_starting_sub", { t: duration(since, u) }), progress: -1, actions: ["engine"], can: false };
  }
  if (jobRunning) {
    const pct = jobPercent(job);
    return { kind: "installing", tone: "busy", card: [name, t("card_download", { pct })], title: t("st_installing_title"),
      sub: job.eta != null ? t("st_installing_sub", { pct, left: duration(job.eta, u) }) : t("st_installing_wait", { pct }),
      progress: pct / 100, actions: ["models"], can: false };
  }
  if (p.state === "stopping") return { kind: "stopping", tone: "idle", card: [name, t("card_stopping")], title: t("eng_stopping"), sub: "", can: false };
  if (p.state === "failed") {
    return { kind: "failed", tone: "bad", card: [name, t("card_failed")], title: t("st_failed_title"),
      sub: errorText(p.error), actions: ["start", "engine"], can: false };
  }
  if (!m) return { kind: "nomodel", tone: "warn", card: ["capy-flash", t("card_setup")], title: t("st_nomodel_title"), sub: t("st_nomodel_sub"), actions: ["models"], can: false };
  if (!m.installed && ["foreign", "different", "outside"].includes(m.state)) {
    // Files are on disk, but not ones the app verified: it neither runs nor changes them without a choice.
    return { kind: "decide", tone: "warn", card: [name, t("card_decide")], title: t("st_decide_title", { model: full }),
      sub: t("st_decide_sub"), actions: ["models"], can: false };
  }
  if (!m.installed) {
    if (job && job.state === "failed") {
      return { kind: "failed", tone: "bad", card: [name, t("card_failed")], title: t("job_failed"), sub: errorText(job.error), actions: ["install", "models"], can: false };
    }
    if (m.partial || (job && job.state === "cancelled")) {
      return { kind: "paused", tone: "warn", card: [name, t("card_paused")], title: t("st_paused_title"), sub: t("st_paused_sub"), actions: ["install", "models"], can: false };
    }
    return { kind: "nomodel", tone: "warn", card: [name, t("card_setup")], title: t("st_nomodel_title"),
      sub: t("eng_not_downloaded", { model: full }), actions: ["models"], can: false };
  }
  return { kind: "stopped", tone: p.error ? "bad" : "idle", card: [name, t("card_stopped")], title: t("st_stopped_title"),
    sub: p.error ? errorText(p.error) : t("st_stopped_sub", { model: full }), actions: ["start", "engine"], can: false };
}

export { bytes };
