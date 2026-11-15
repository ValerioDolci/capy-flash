// capy-flash app — the panel: engine or connection, models, tuning, tools, settings. One modal window with its
// sections on the left; details that few people need sit under "Advanced".
import { $, el, icon, api, copyText, num, bytes, duration, tilde } from "/util.js";
import { t, units, applyLanguage } from "/i18n.js";
import { S, isLocal, isMac, status, act, loadEngine, loadModels, loadState, watchEngine, chosenModel, modelName,
  jobPercent, errorText, checkText, changed, chatModels, busyNames } from "/state.js";

const P = { section: "", setup: null, plan: null, long: false, update: null, log: null, note: {}, signature: "" };
let hooks = { startSetup: () => {}, applyTheme: () => {}, afterBackendChange: async () => {}, quit: () => {} };
export const setPanelHooks = (h) => { hooks = Object.assign(hooks, h); };

const sections = () => (isLocal() ? ["engine", "models", "tuning", "tools", "settings"] : ["connection", "tools", "settings"]);

export function openPanel(section) {
  const list = sections();
  if (section === "engine" && !isLocal()) section = "connection";
  if (section === "connection" && isLocal()) section = "engine";
  P.section = list.includes(section) ? section : list[0];
  const dialog = $("panel");
  if (!dialog.open) { dialog.showModal(); watchEngine(true); }
  P.signature = "";
  enter();
  render(true);
  $("panel-body").scrollTop = 0;
}

export function closePanel() { if ($("panel").open) $("panel").close(); }

/** Things a section needs from the server when it opens. */
function enter() {
  if (P.section === "connection" || P.section === "engine") {
    api("GET", "/api/setup").then((s) => { P.setup = s; render(); }).catch(() => {});
  }
  if (P.section === "tuning") loadPlan();
  if (P.section === "tools") fillTools();
  loadEngine();
}

function loadPlan() {
  const key = S.eng && S.eng.model;
  if (!key) return;
  api("GET", `/api/tune/plan?model=${encodeURIComponent(key)}&long=${P.long ? 1 : 0}`).then((p) => { P.plan = p; render(); }).catch(() => {});
}

/** Rebuild the open section when what it shows changed; never under the user's cursor in a text field. */
export function render(force) {
  const dialog = $("panel");
  if (!dialog.open) return;
  const body = $("panel-body");
  const active = document.activeElement;
  const typing = body.contains(active) && ["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName) && active.type !== "checkbox";
  const signature = JSON.stringify([P.section, S.eng, S.settings, S.modelsError, S.models.length, P.setup, P.plan, P.long,
    P.update, P.log, P.note, S.opencode, S.folder]);
  if (!force && (signature === P.signature || typing)) return;
  P.signature = signature;
  const openKeys = [...body.querySelectorAll("details[open][data-k]")].map((d) => d.dataset.k);
  const focusKey = body.contains(active) && active.dataset ? active.dataset.k : null;
  const scroll = body.scrollTop;

  const nav = $("panel-nav");
  nav.textContent = "";
  for (const s of sections()) {
    if (s === "tools") nav.append(el("div", "nav-sep"));
    nav.append(el("button", { class: "nav-item", type: "button", "aria-current": s === P.section ? "page" : null,
      onclick: () => { P.section = s; P.note = {}; enter(); render(true); body.scrollTop = 0; } }, t("nav_" + s)));
  }
  $("panel-title").textContent = t("nav_" + P.section);
  body.textContent = "";
  body.append(...{ engine, models, tuning, connection, tools, settings }[P.section]().filter(Boolean));
  for (const k of openKeys) { const d = body.querySelector(`details[data-k="${k}"]`); if (d) d.open = true; }
  body.scrollTop = scroll;
  if (focusKey) { const f = body.querySelector(`[data-k="${focusKey}"]`); if (f) f.focus(); }
}

// ---- building blocks ----------------------------------------------------------------------------------------------
const h3 = (text) => el("h3", null, text);
const note = (text, tone) => el("p", { class: "note" + (tone ? " " + tone : ""), role: tone === "bad" ? "alert" : null }, text);
const field = (label, input, hint) => el("label", "field", el("span", "field-label", label, hint ? el("span", "field-hint", hint) : null), input);
function button(label, opts = {}) {
  return el("button", { class: "btn" + (opts.primary ? " btn-primary" : "") + (opts.quiet ? " btn-quiet" : "") + (opts.danger ? " btn-danger" : "") + (opts.small ? " btn-small" : ""),
    type: "button", "data-k": opts.k, disabled: opts.disabled, onclick: opts.onclick, title: opts.title }, opts.icon ? icon(opts.icon) : null, el("span", null, label));
}
function details(key, summary, ...children) {
  return el("details", { class: "more", "data-k": key }, el("summary", null, icon("chevron"), el("span", null, summary)), el("div", "more-body", ...children));
}
export function codeBlock(text, key) {
  const copy = el("button", { class: "copy", type: "button", "data-k": key }, el("span", null, t("copy")));
  copy.addEventListener("click", () => copyText(text, copy, t("copied")));
  return el("div", "snippet", copy, el("pre", null, text));
}
export const tunnel = () => codeBlock("ssh -N -L 8080:127.0.0.1:8080 user@your-linux-machine", "tunnel");
/** Run an action; a refusal is shown under the section, in the user's words when the app knows the reason. */
function run(key, fn) {
  return async (ev) => {
    const b = ev && ev.currentTarget;
    if (b) b.disabled = true;
    P.note = {};
    try { await fn(); } catch (e) { P.note = { [key]: errorText({ code: e.code, message: e.message }) || e.message }; }
    render(true);
  };
}
const noteFor = (key) => (P.note[key] ? note(P.note[key], "bad") : null);

export function checkList(lines) {
  return el("ul", "checks", lines.map((l) => el("li", "check is-" + l.level, icon(l.level === "ok" ? "check" : "warn"), el("span", null, checkText(l)))));
}

export function machineLines(facts) {
  const rows = [];
  const usable = facts.gpus.filter((g) => g.usable);
  const byName = {};
  for (const g of facts.gpus) byName[g.name] = (byName[g.name] || 0) + 1;
  if (!facts.gpus.length) rows.push(["GPU", t("gpu_none")]);
  for (const [name, n] of Object.entries(byName)) {
    const g = facts.gpus.find((x) => x.name === name);
    rows.push(["GPU", `${n}× ${name.replace(/^NVIDIA GeForce /, "")} · ${Math.round(g.memory_mib / 1024)} GB`, usable.some((x) => x.name === name)]);
  }
  if (facts.cpu.model) rows.push(["CPU", `${facts.cpu.model} · ${t("cpu_cores", { n: facts.cpu.physical_cores })}` + (facts.cpu.simd ? ` · ${facts.cpu.simd === "avx512" ? "AVX-512" : "AVX2"}` : "")]);
  if (facts.ram_total_gib) rows.push(["RAM", t("ram_line", { total: facts.ram_total_gib, free: facts.ram_available_gib })]);
  if (facts.disk_free_gb != null) rows.push(["Disk", t("disk_line", { free: num(facts.disk_free_gb) })]);
  return el("dl", "facts", rows.map(([k, v]) => [el("dt", null, k), el("dd", null, v)]));
}

export function progressBlock(job, opts = {}) {
  const u = units();
  const pct = jobPercent(job);
  const bar = el("div", { class: "bar-track", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(pct) }, el("span"));
  bar.firstChild.style.width = pct + "%";
  const what = job.phase === "release" ? t("job_release", { v: job.version || "" }).trim() : t("job_model");
  const step = t("job_" + (["download", "verify", "unpack"].includes(job.step) ? job.step : "prepare"));
  const parts = [`${what} — ${step.toLowerCase()}`];
  if (job.bytes_total) parts.push(t("job_progress", { done: bytes(job.bytes_done), total: bytes(job.bytes_total) }));
  if (job.step === "download" && job.speed > 0) parts.push(t("job_speed", { speed: bytes(job.speed) }));
  if (job.step === "download" && job.eta != null) parts.push(t("job_left", { left: duration(job.eta, u) }));
  if (job.step === "verify" && job.step_total) parts.push(Math.floor((job.step_done / job.step_total) * 100) + " %");
  return el("div", "progress", bar, el("div", "progress-row", el("span", "progress-text", parts.join(" · ")),
    opts.noCancel ? null : button(t("job_cancel"), { small: true, quiet: true, k: "job-cancel", onclick: run("models", () => act("/api/engine/cancel", { job: job.id })) })));
}

// ---- Engine -------------------------------------------------------------------------------------------------------
function engine() {
  const e = S.eng;
  if (!e) return [note(t("conn_checking"))];
  const u = units();
  const p = e.process, m = chosenModel(), st = status();
  const out = [];
  const label = { running: "eng_running", starting: "eng_starting", stopping: "eng_stopping", failed: "eng_failed", stopped: "eng_stopped" }[p.state];
  const tone = { running: "ok", starting: "busy", stopping: "busy", failed: "bad", stopped: "idle" }[p.state];
  const meta = [];
  if (m) meta.push(`${m.title} · ${m.edition}`);
  if (p.state === "running" || p.state === "starting") {
    meta.push(t("eng_meta_gpus", { n: p.gpus }), t("eng_meta_port", { port: p.port }));
    if (p.load_seconds) meta.push(t("eng_loaded", { t: duration(p.load_seconds, u) }));
    if (p.restarts) meta.push(t("eng_restarts", { n: p.restarts }));
    if (p.adopted) meta.push(t("eng_adopted"));
  }
  const tuning = e.tuning.state === "running";
  const canStart = m && m.installed && e.release && !tuning;
  let action = null;
  if (p.state === "running" || p.state === "starting") action = button(t("eng_stop"), { k: "eng-stop", onclick: run("engine", () => act("/api/engine/stop")) });
  else if (p.state !== "stopping") action = button(t(p.state === "failed" ? "eng_start_again" : "eng_start"), { primary: true, k: "eng-start", disabled: !canStart, onclick: run("engine", () => act("/api/engine/start")) });
  out.push(el("div", "state-row", el("span", "dot " + tone), el("div", "state-text", el("strong", null, tuning ? t("st_tuning_title") : t(label)), el("span", null, meta.join(" · "))), action));
  out.push(noteFor("engine"));
  if (!m) out.push(note(t("eng_no_model")), button(t("choose_model"), { primary: true, onclick: () => go("models") }));
  else if (!m.installed && !(e.job && e.job.state === "running")) out.push(note(t("eng_not_downloaded", { model: modelName(m) })), button(t("nav_models"), { onclick: () => go("models") }));
  if (e.job && e.job.state === "running") out.push(progressBlock(e.job));
  if (p.state === "failed" && p.error) {
    out.push(el("div", { class: "fail", role: "alert" }, el("strong", null, errorText(p.error) || p.error.message),
      p.error.log && p.error.log.length ? el("pre", "log", p.error.log.join("\n")) : null));
  } else if (p.state === "stopped" && p.error) {
    out.push(el("div", { class: "fail", role: "alert" }, el("strong", null, errorText(p.error))));
  }
  if (e.release && e.release.missing_libs && e.release.missing_libs.length) {
    out.push(el("div", "warn-box", el("strong", null, t("eng_missing_libs", { libs: e.release.missing_libs.join(", ") })), el("span", null, t("eng_missing_libs_fix")),
      codeBlock("sudo apt install libavcodec-dev libavformat-dev libswscale-dev libcurl4", "apt")));
  }
  out.push(h3(t("eng_this_computer")), machineLines(e.machine));

  // Advanced: everything a curious or careful user may want, out of the way
  const port = el("input", { class: "short", inputmode: "numeric", value: String(e.settings.port), "data-k": "port", "aria-label": t("eng_port") });
  port.addEventListener("change", run("adv", () => act("/api/engine/settings", { port: parseInt(port.value, 10) })));
  const auto = el("input", { type: "checkbox", checked: e.settings.autostart, "data-k": "autostart" });
  auto.addEventListener("change", run("adv", () => act("/api/engine/settings", { autostart: auto.checked })));
  const adv = [el("div", "field-row", field(t("eng_port"), port), el("label", "check-field", auto, el("span", null, t("eng_autostart")))), noteFor("adv")];
  adv.push(el("p", "kv", el("span", null, t("eng_folder", { dir: tilde(e.dir, S.home) })), el("span", "muted", t("eng_folder_hint"))));
  if (e.release) {
    const upd = P.update;
    const line = el("p", "kv", el("span", null, t("eng_release", { v: e.release.version }) + " · " + t(e.release.verified ? "verified" : "not_verified")));
    if (upd && upd.newer) line.append(button(t("eng_update_to", { v: upd.latest }), { small: true, k: "update", onclick: run("adv", async () => { await act("/api/engine/update"); P.update = null; }) }));
    else if (upd) line.append(el("span", "muted", t("eng_up_to_date")));
    else line.append(button(t("eng_check_update"), { small: true, quiet: true, k: "check-update", onclick: run("adv", async () => { P.update = await api("POST", "/api/engine/check-update", {}); }) }));
    adv.push(line);
  }
  if (e.command) adv.push(el("p", "field-label", t("eng_command")), codeBlock(e.command.split(S.home + "/").join("~/"), "cmd"));
  const logBox = details("log", t("eng_log"), P.log == null ? note(t("conn_checking")) : P.log.length ? el("pre", "log", P.log.join("\n")) : note(t("eng_log_empty")));
  logBox.addEventListener("toggle", () => { if (logBox.open) api("GET", "/api/engine/log?lines=200").then((r) => { P.log = r.lines; render(); }).catch(() => {}); });
  adv.push(logBox);
  adv.push(details("lossy", t("eng_lossy_title"), note(t("eng_lossy_text")),
    el("table", "plain", el("thead", null, el("tr", null, el("th", null, t("lossy_switch")), el("th", null, t("lossy_gain")), el("th", null, t("lossy_loss")))),
      el("tbody", null, e.lossy.map((s) => el("tr", null, el("td", null, el("code", null, `${s.env}=${s.value}`)), el("td", null, s.gain), el("td", null, s.loss)))))));
  out.push(details("adv", t("advanced"), ...adv));
  out.push(details("server", t("eng_to_server"), note(t("conn_stop_note")), ...connectionForm()));
  return out;
}

// ---- Models -------------------------------------------------------------------------------------------------------
function modelFacts(m) {
  return t("m_facts", { size: bytes(m.size_gb * 1e9), gpus: m.gpus, ram: m.need_ram_gib, ctx: Math.floor(m.context / 1000) }) + (m.vision ? " · " + t("m_images") : "");
}

/** What is on disk for a model that the app did not verify, and what the app will (not) do about it. */
function onDiskNote(m) {
  const folder = tilde(m.folder, S.home);
  if (m.state === "foreign") return el("p", "card-note disk-note", t("m_found", { folder }));
  if (m.state === "different") {
    return el("div", "warn-box disk-note", el("span", null, t("m_different", { folder })),
      m.mismatch && m.mismatch.length ? el("span", "muted small", t("m_different_files", { files: m.mismatch.join(", ") })) : null);
  }
  if (m.state === "outside") return el("div", "warn-box disk-note", el("span", null, t("m_outside", { folder })));
  if (m.state === "as_is") return el("p", "card-note disk-note", t("m_as_is_state", { folder }));
  return null;
}

export function modelCard(m, e, opts = {}) {
  const job = e.job && e.job.model === m.key ? e.job : null;
  const inUse = e.mode === "local" && e.model === m.key;
  const badges = [];
  if (inUse && m.installed) badges.push(el("span", "badge on", t("m_in_use")));
  if (m.recommended && !inUse) badges.push(el("span", "badge", t("m_recommended")));
  const card = el("div", "card model" + (inUse ? " current" : "") + (m.fit.ok ? "" : " unfit"),
    el("div", "card-head", el("strong", null, m.title), el("span", "edition", m.edition), ...badges),
    el("p", "card-note", t("m_note_" + m.key)), el("p", "card-facts", modelFacts(m)));
  const problems = m.fit.lines.filter((l) => l.level !== "ok");
  if (problems.length) card.append(checkList(problems));
  const running = job && job.state === "running";
  if (m.fit.ok && !running) { const disk = onDiskNote(m); if (disk) card.append(disk); }
  if (!m.fit.ok || opts.readOnly) return card;
  const row = el("div", "card-actions");
  const size = bytes(m.size_gb * 1e9);
  const busy = e.job && e.job.state === "running";
  const go = (path, body) => run("m-" + m.key, () => act(path, Object.assign({ model: m.key }, body)));
  const useButton = () => button(t("m_use"), { primary: true, k: "use-" + m.key, onclick: go("/api/engine/use") });
  const asIs = () => button(t("m_as_is"), { small: true, quiet: true, k: "asis-" + m.key, onclick: go("/api/engine/use", { as_is: true }) });
  const newFolder = (primary) => button(t("m_new_folder", { size }), { primary, small: !primary, quiet: !primary, k: "new-" + m.key, disabled: busy, onclick: go("/api/engine/install", { new_folder: true }) });
  if (running) {
    card.append(progressBlock(job));
  } else if (m.state === "ready") {
    const elsewhere = !m.folder.endsWith("/models/" + m.key);  // downloaded next to a folder the app did not create
    row.append(el("span", "ok-text", icon("check"), !m.deletable ? t("m_verified_found", { folder: tilde(m.folder, S.home) })
      : elsewhere ? t("m_downloaded_in", { folder: tilde(m.folder, S.home) }) : t("m_downloaded")));
    if (!inUse) row.append(useButton());
    const engineOn = inUse && ["running", "starting", "stopping"].includes(e.process.state);
    if (m.deletable && !engineOn) {  // only what the app downloaded, and not while the engine runs it
      const del = button(t("m_delete"), { small: true, quiet: true, k: "del-" + m.key });
      let armed = false;
      del.addEventListener("click", async (ev) => {
        if (!armed) { armed = true; del.classList.add("btn-danger"); del.querySelector("span").textContent = t("m_delete_confirm", { size: bytes(m.bytes) }); setTimeout(() => { armed = false; render(true); }, 4000); return; }
        await go("/api/engine/remove")(ev);
      });
      row.append(del);
    }
  } else if (m.state === "as_is") {
    if (!inUse) row.append(useButton());
    if (m.found !== "outside") row.append(button(t("m_verify"), { small: true, quiet: true, k: "verify-" + m.key, disabled: busy, onclick: go("/api/engine/install") }));
    row.append(newFolder(false));
  } else if (m.state === "foreign") {
    row.append(button(t("m_verify_size", { size: bytes(m.bytes) }), { primary: true, k: "verify-" + m.key, disabled: busy, onclick: go("/api/engine/install") }), asIs());
  } else if (m.state === "different" || m.state === "outside") {
    row.append(newFolder(true), asIs());
  } else {
    if (job && job.state === "failed") card.append(el("div", { class: "fail", role: "alert" }, el("strong", null, t("job_failed")), el("span", "fail-detail", errorText(job.error))));
    else if (m.partial || (job && job.state === "cancelled")) card.append(note(t("job_cancelled")));
    const label = m.partial ? t("m_continue", { size }) : job && job.state === "failed" ? t("try_again") : t("m_download", { size });
    row.append(button(label, { primary: m.recommended || inUse || m.partial, k: "get-" + m.key, disabled: busy, onclick: go("/api/engine/install") }));
  }
  if (row.childNodes.length) card.append(row);
  const refusal = noteFor("m-" + m.key);
  if (refusal) card.append(refusal);
  return card;
}

function models() {
  const e = S.eng;
  if (!e) return [note(t("conn_checking"))];
  const fit = e.models.filter((m) => m.fit.ok), unfit = e.models.filter((m) => !m.fit.ok);
  const rank = (m) => (e.model === m.key ? 0 : m.installed ? 1 : m.recommended ? 2 : m.optional ? 4 : 3);
  fit.sort((a, b) => rank(a) - rank(b));
  const out = [note(t("models_lede")), noteFor("models")];
  if (e.job && e.job.state === "running" && e.job.phase === "release" && !e.job.model) out.push(progressBlock(e.job));
  out.push(el("div", "cards", fit.map((m) => modelCard(m, e))));
  if (unfit.length) out.push(details("unfit", `${t("m_notfit")} (${unfit.length})`, el("div", "cards", unfit.map((m) => modelCard(m, e)))));
  return out;
}

// ---- Tuning -------------------------------------------------------------------------------------------------------
const METRICS = ["decode_tps", "prefill_tps", "short_ms", "long_decode_tps", "long_prefill_tps"];
const metricValue = (k, v) => (k === "short_ms" ? `${num(v)} ms` : `${num(v, v < 100 ? 1 : 0)} ${t("tok_s")}`);

/** The kept settings in words, in the order of the tuner's table (GET /api/engine: lossless). */
function flagWords(flags, settings) {
  return settings.filter((s) => flags[s.key] != null && flags[s.key] !== false).map((s) => t("f_" + s.id, { v: flags[s.key] }));
}

export function beforeAfter(before, after) {
  const cards = [];
  for (const k of METRICS) {
    if (after[k] == null) continue;
    const b = before[k], a = after[k];
    const same = b == null || b === a;
    const change = same ? 0 : k === "short_ms" ? (b - a) / b : (a - b) / b;
    cards.push(el("div", "metric", el("span", "metric-name", t("met_" + k)), el("span", "metric-value", metricValue(k, a)),
      same ? el("span", "metric-delta flat", b == null ? "" : t("tune_same"))
        : el("span", "metric-delta " + (change > 0 ? "up" : "down"), `${a > b ? "+" : "−"}${num(Math.abs(a - b) / b * 100, 0)} % · ${t("tune_before").toLowerCase()} ${metricValue(k, b)}`)));
  }
  return el("div", "metrics", cards);
}

/** How far apart the two measurements of the defaults were: what a setting has to beat, besides the 3 %. */
function noiseLine(noise) {
  const parts = METRICS.filter((k) => noise && noise[k] != null).map((k) => `${t("met_" + k).toLowerCase()} ±${num(noise[k] * 100, 1)} %`);
  return parts.length ? el("p", "muted small", t("tune_noise", { values: parts.join(" · ") })) : null;
}

function trialsTable(trials) {
  const head = el("tr", null, el("th", null, t("tune_col_test")), ...METRICS.slice(0, 3).map((k) => el("th", "n", t("met_" + k))));
  const rows = trials.map((x) => {
    const name = t("step_" + x.step) + (x.label ? ` — ${x.label}` : "");
    const cells = x.metrics ? METRICS.slice(0, 3).map((k) => el("td", "n", x.metrics[k] != null ? metricValue(k, x.metrics[k]) : ""))
      : [el("td", { class: "n muted", colspan: "3" }, x.error ? t("tune_trial_failed") : t("tune_trial_running"))];
    return el("tr", x.adopted ? "adopted" : null, el("td", null, name, x.adopted ? el("span", "badge on", t("tune_adopted")) : null), ...cells);
  });
  return el("div", "table-wrap", el("table", "plain", el("thead", null, head), el("tbody", null, rows)));
}

function tuning() {
  const e = S.eng;
  if (!e) return [note(t("conn_checking"))];
  const u = units();
  const m = chosenModel(), tune = e.tuning, profile = e.profile, plan = P.plan && P.plan.model === e.model ? P.plan : null;
  if (!m || !m.installed) return [note(t("tune_lede")), note(t("tune_need_model")), button(t("nav_models"), { onclick: () => go("models") })];
  const out = [];
  if (tune.state === "running") {
    const n = Math.min(tune.done + 1, tune.total);
    const bar = el("div", { class: "bar-track", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": String(tune.total), "aria-valuenow": String(tune.done) }, el("span"));
    bar.firstChild.style.width = Math.round((tune.done / Math.max(tune.total, 1)) * 100) + "%";
    const what = t("step_" + (tune.step || "baseline")) + (tune.label ? ` — ${tune.label}` : "");
    out.push(el("div", "progress", el("div", "progress-title", el("strong", null, t("tune_running", { n, m: tune.total })), el("span", null, what)), bar,
      el("div", "progress-row", el("span", "progress-text", tune.eta != null ? t("tune_left", { t: duration(tune.eta, u) }) : ""),
        button(t("tune_cancel"), { small: true, k: "tune-cancel", onclick: run("tuning", () => act("/api/tune/cancel")) }))));
    out.push(note(t("tune_time", { t: duration(tune.eta || 0, u) }).split(". ").slice(1).join(". ")));
    if (tune.trials.length) out.push(trialsTable(tune.trials));
    return out;
  }
  out.push(note(t("tune_lede")));
  if (tune.state === "failed") out.push(note(t("tune_failed", { e: tune.error || "" }), "bad"));
  if (tune.state === "cancelled") out.push(note(t("tune_cancelled")));
  out.push(noteFor("tuning"));
  const tunable = plan ? plan.tunable : m.family === "flash-next";
  if (profile && profile.after) {
    if (profile.stale) out.push(note(t("tune_stale"), "warn"));
    out.push(beforeAfter(profile.before || {}, profile.after));
    out.push(noiseLine(profile.noise));
    const words = flagWords(profile.flags || {}, e.lossless);
    const kept = !tunable ? t("tune_measured_only") : words.length ? null : t("tune_nothing");
    if (words.length) out.push(el("div", "kept" + (profile.active ? "" : " off"), el("span", "kept-label", t("tune_kept")), ...words.map((w) => el("span", "chip", w))));
    else out.push(note(kept));
    if (!profile.active && words.length) out.push(note(t("tune_reverted")));
    const when = new Date((profile.measured_at || 0) * 1000).toLocaleDateString(document.documentElement.lang, { day: "numeric", month: "long" });
    out.push(el("p", "muted small", t("tune_measured", { date: when, t: duration(profile.seconds || 0, u), v: profile.engine_version || "?" })));
    const row = el("div", "actions");
    if (words.length && profile.active) row.append(button(t("tune_revert"), { k: "revert", onclick: run("tuning", () => act("/api/tune/revert")) }));
    if (words.length && !profile.active && !profile.stale) row.append(button(t("tune_apply"), { primary: true, k: "apply", onclick: run("tuning", () => act("/api/tune/apply")) }));
    row.append(button(t(tunable ? "tune_again" : "tune_measure_again"), { quiet: true, k: "tune", onclick: startTuning }));
    out.push(row);
    if (plan && tunable) out.push(longOption(plan, u));
    if (profile.trials && profile.trials.length > 1) out.push(details("trials", t("tune_details"), trialsTable(profile.trials)));
    return out;
  }
  if (!plan) return out.concat([note(t("conn_checking"))]);
  if (!tunable) {
    out.push(note(t("tune_dense", { t: duration(plan.seconds, u) })), el("div", "actions", button(t("tune_measure"), { primary: true, k: "tune", onclick: startTuning })));
    return out;
  }
  out.push(el("div", "card tune-card",
    el("p", "card-note", t("tune_time", { t: duration(plan.seconds, u) })),
    el("ul", "steps", plan.steps.map((s) => el("li", null, t("step_" + s.id)))),
    longOption(plan, u),
    el("div", "card-actions", button(t("tune_start"), { primary: true, k: "tune", onclick: startTuning }), el("span", "muted small", t("tune_lossless")))));
  return out;
}

function longOption(plan, u) {
  const box = el("input", { type: "checkbox", checked: P.long, "data-k": "long" });
  box.addEventListener("change", () => { P.long = box.checked; loadPlan(); });
  return el("label", "check-field", box, el("span", null, t("tune_long", { t: duration(plan.long_extra_seconds || 120, u) })));
}

const startTuning = run("tuning", () => act("/api/tune/start", { long_context: P.long }));

// ---- Connection ---------------------------------------------------------------------------------------------------
function connectionForm() {
  const b = S.settings.backend;
  const name = el("input", { value: b.name || "", autocomplete: "off", "data-k": "c-name" });
  const url = el("input", { value: isLocal() ? "" : b.base_url, autocomplete: "off", spellcheck: "false", placeholder: "http://127.0.0.1:8080/v1", "data-k": "c-url" });
  const key = el("input", { type: "password", autocomplete: "off", placeholder: b.has_api_key ? t("conn_key_kept") : t("conn_key_none"), "data-k": "c-key" });
  const save = button(t("conn_save"), { primary: true, k: "c-save", onclick: run("conn", () => useServer(name.value.trim() || "server", url.value.trim(), key.value || null)) });
  return [el("div", "form server-form", field(t("conn_name"), name), field(t("conn_url"), url), field(t("conn_key"), key),
    el("div", "actions", save)), noteFor("conn")];
}

export async function useServer(name, baseUrl, apiKey) {
  S.settings = await api("POST", "/api/settings", { backend: { name, base_url: baseUrl, api_key: apiKey } });
  await loadState();
  await loadModels();
  await hooks.afterBackendChange();
  P.setup = null;
  api("GET", "/api/setup").then((s) => { P.setup = s; render(); }).catch(() => {});
}

function foundServers() {
  const s = P.setup;
  if (!s) return [];
  const b = S.settings.backend;
  const rows = s.endpoints.map((e) => {
    const current = !isLocal() && e.base_url === b.base_url;
    return el("div", "found", el("div", "found-text", el("strong", null, e.name), el("span", "muted", e.base_url.replace(/^https?:\/\//, "") + " · " + t("conn_models_n", { n: e.models }))),
      current ? el("span", "badge on", t("conn_current")) : button(t("conn_use"), { small: true, k: "use-" + e.base_url, onclick: run("conn", () => useServer(e.name, e.base_url, null)) }));
  });
  const look = button(t("conn_look"), { small: true, quiet: true, k: "look", onclick: () => { P.setup = null; render(true); api("GET", "/api/setup").then((x) => { P.setup = x; render(); }); } });
  return [el("div", "h3-row", h3(t("conn_found")), look), rows.length ? el("div", "found-list", rows) : note(t("conn_none"))];
}

function connection() {
  const b = S.settings.backend, st = status();
  const out = [];
  const head = st.kind === "ready" ? t("conn_ok", { name: b.name, n: chatModels().length }) : st.kind === "checking" ? t("conn_checking") : t("conn_bad", { name: b.name });
  out.push(el("div", "state-row", el("span", "dot " + st.tone), el("div", "state-text", el("strong", null, head),
    el("span", null, st.kind === "unreachable" ? S.modelsError : b.base_url)), st.kind === "ready" ? null : button(t("try_again"), { small: true, k: "c-retry", onclick: () => loadModels().then(() => render(true)) })));
  out.push(...connectionForm());
  out.push(...foundServers());
  const s = P.setup;
  if (isMac()) out.push(el("div", "info-box", el("p", null, t("conn_mac")), el("p", null, t("conn_tunnel")), tunnel()));
  else if (s && !s.supported) out.push(el("div", "info-box", el("p", null, t("conn_unsupported")), checkList(s.why_not), el("p", null, t("conn_tunnel")), tunnel()));
  else if (s && s.supported) {
    const busy = s.busy_gpus || [];
    out.push(el("div", "card local-card", el("div", "card-head", el("strong", null, t("conn_local_title"))),
      el("p", "card-note", t("conn_local_text")),
      busy.length ? note(t("conn_local_busy", { names: busyNames(busy) }), "warn") : null,
      el("div", "card-actions", button(t("conn_local_btn"), { k: "local-setup", onclick: () => { closePanel(); hooks.startSetup("model"); } }))));
  }
  return out;
}

// ---- Tools --------------------------------------------------------------------------------------------------------
const T = { snippets: null, listing: null, result: "", manual: "", launched: "" };

async function fillTools() {
  T.snippets = null;
  try { T.snippets = await api("GET", "/api/snippets?model=" + encodeURIComponent($("model").value || "")); } catch (_) { T.snippets = {}; }
  await browse(S.folder || "~");
}

async function browse(path) {
  try { T.listing = await api("GET", "/api/folders?path=" + encodeURIComponent(path)); S.folder = T.listing.path; }
  catch (e) { T.listing = { path, dirs: [], error: e.message }; }
  render(true);
}

function tools() {
  const oc = S.opencode || {};
  const out = [note(t("tools_lede")), h3("OpenCode")];
  if (!oc.installed) {
    out.push(note(t("oc_missing")));
  } else {
    out.push(note(t("oc_text")));
    if (!oc.supported) out.push(note(t("oc_unsupported", { v: oc.version }), "warn"));
    const l = T.listing;
    const list = el("div", "picker-list");
    if (l && l.parent) list.append(el("button", { class: "picker-item up", type: "button", onclick: () => browse(l.parent) }, "↑ " + t("folder_up")));
    for (const d of (l && l.dirs) || []) list.append(el("button", { class: "picker-item", type: "button", onclick: () => browse(l.path.replace(/\/$/, "") + "/" + d) }, d));
    out.push(el("div", "field", el("span", "field-label", t("project_folder")),
      el("div", "picker", el("div", "picker-path", l ? l.path + (l.git ? "  ·  " + t("git") : "") : ""), list)));
    const recent = (S.settings.recent_folders || []).map((f) => el("button", { class: "chip", type: "button", title: f, onclick: () => browse(f) }, f.split("/").filter(Boolean).pop() || f));
    if (recent.length) out.push(el("div", "chips", recent));
    out.push(el("div", "actions", button(t("oc_launch"), { primary: true, k: "oc-launch", disabled: !chatModels().length, onclick: run("oc", async () => {
      const r = await api("POST", "/api/opencode/configure", { model: $("model").value });
      T.manual = r.manual;
      await api("POST", "/api/opencode/launch", { folder: S.folder, model: $("model").value });
      T.launched = t("oc_launched");
      await loadState();
    }) }), el("span", "muted small", t("oc_found", { v: oc.version || "" }))));
    if (T.launched) out.push(note(T.launched, "ok"));
    out.push(noteFor("oc"));
    const manual = details("oc-manual", t("oc_manual"), T.manual ? codeBlock(T.manual, "oc-manual-copy") : note(t("conn_checking")));
    manual.addEventListener("toggle", async () => {
      if (!manual.open || T.manual) return;
      try { T.manual = (await api("POST", "/api/opencode/configure", { model: $("model").value })).manual; } catch (e) { T.manual = e.message; }
      render(true);
    });
    out.push(manual);
  }
  out.push(h3(t("snippets")));
  const sn = T.snippets || {};
  for (const [name, label] of [["claude_code", "Claude Code"], ["python", "Python (openai)"], ["curl", "curl"]]) {
    if (sn[name]) out.push(el("p", "field-label", label), codeBlock(sn[name], "sn-" + name));
  }
  return out;
}

// ---- Settings -----------------------------------------------------------------------------------------------------
function settings() {
  const s = S.settings;
  const save = async (change) => {
    S.settings = await api("POST", "/api/settings", change);
    hooks.applyTheme();
    applyLanguage(S.settings.language);
    changed();
    render(true);
  };
  const temp = el("input", { class: "short", inputmode: "decimal", value: s.temperature == null ? "" : String(s.temperature), "data-k": "temp" });
  temp.addEventListener("change", () => { const v = temp.value.trim().replace(",", "."); save({ temperature: v === "" || isNaN(Number(v)) ? null : Number(v) }); });
  const max = el("input", { class: "short", inputmode: "numeric", value: s.max_tokens == null ? "" : String(s.max_tokens), "data-k": "max" });
  max.addEventListener("change", () => { const v = parseInt(max.value, 10); save({ max_tokens: v > 0 ? v : null }); });
  const seg = (current, options, onpick, key) => el("div", { class: "seg", role: "group" }, options.map(([value, label]) =>
    el("button", { type: "button", class: "seg-item", "aria-pressed": current === value ? "true" : "false", "data-k": key + value, onclick: () => onpick(value) }, label)));
  const mod = /Mac/.test(navigator.platform) ? "⌘" : "Ctrl";
  const keys = [["Enter", "sc_send"], ["Shift + Enter", "sc_newline"], ["Esc", "sc_stop"], [`${mod} + Shift + O`, "sc_new"], ["/", "sc_focus"]];
  return [
    h3(t("generation")),
    el("div", "field-row", field(t("temperature"), temp, t("default_hint")), field(t("max_tokens"), max, t("default_hint"))),
    h3(t("appearance")),
    el("div", "field-row", el("div", "field", el("span", "field-label", t("theme")),
      seg(s.theme, [["system", t("theme_system")], ["dark", t("dark")], ["light", t("light")]], (v) => save({ theme: v }), "theme-")),
    el("div", "field", el("span", "field-label", t("language")),
      seg(s.language === "en" || s.language === "it" ? s.language : "auto", [["auto", t("lang_auto")], ["en", "English"], ["it", "Italiano"]], (v) => save({ language: v }), "lang-"))),
    h3(t("keyboard")),
    el("dl", "keys", keys.map(([k, label]) => [el("dt", null, el("kbd", null, k)), el("dd", null, t(label))])),
    h3(t("about")),
    el("div", "about", el("span", "muted", t("version", { v: S.version })),
      button(t(isLocal() && S.eng && ["running", "starting"].includes(S.eng.process.state) ? "quit_local" : "quit"), { k: "quit", onclick: hooks.quit })),
  ];
}

function go(section) { P.section = section; P.note = {}; enter(); render(true); $("panel-body").scrollTop = 0; }

export function initPanel() {
  const dialog = $("panel");
  $("close-panel").addEventListener("click", closePanel);
  dialog.addEventListener("close", () => { watchEngine(false); P.note = {}; $("input").focus(); });
  dialog.addEventListener("click", (e) => { if (e.target === dialog) closePanel(); });  // a click on the veil
}
