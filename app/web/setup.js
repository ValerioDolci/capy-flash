// capy-flash app — the first run: one decision per screen. Where do the models run? → which model? → install →
// start and (optionally) tune. On a Mac, or with a server already answering, it is a single screen.
import { $, el, icon, api, bytes, duration, tilde } from "/util.js";
import { t, units } from "/i18n.js";
import { S, isMac, act, loadEngine, loadModels, loadState, watchEngine, chosenModel, modelName, errorText, busyNames } from "/state.js";
import { checkList, machineLines, modelCard, progressBlock, useServer, openPanel, tunnel } from "/panel.js";

const U = { open: false, step: "where", info: null, pick: "", server: "", note: "", plan: null, from: "", signature: "" };
let onFinish = () => {};
export const setSetupHooks = (h) => { onFinish = h.onFinish; };
export const setupOpen = () => U.open;

export async function startSetup(step) {
  U.open = true; U.step = step || "where"; U.note = ""; U.info = null; U.signature = ""; U.from = step ? "panel" : "";
  $("setup").hidden = false;
  $("shell").hidden = true;
  watchEngine(true);
  renderSetup(true);
  const [info] = await Promise.all([api("GET", "/api/setup"), loadEngine()]);
  U.info = info;
  if (!step) U.step = info.proposal === "connect" ? "connect" : "where";
  if (info.endpoints.length) {
    const current = info.endpoints.find((e) => e.base_url === S.settings.backend.base_url);
    U.server = (current || info.endpoints[0]).base_url;
  }
  U.pick = (S.eng && S.eng.model) || info.recommended || "";
  renderSetup(true);
}

async function finish(openSection) {
  if (!S.settings.setup_done) { try { await api("POST", "/api/setup/done", {}); } catch (_) { /* the app is closing */ } }
  await loadState();
  U.open = false;
  $("setup").hidden = true;
  $("shell").hidden = false;
  watchEngine(false);
  await onFinish();
  if (openSection) openPanel(openSection);
}

const title = (text) => el("h1", { tabindex: "-1" }, text);
const lede = (text) => el("p", "lede", text);
const btn = (label, opts = {}) => el("button", { class: "btn" + (opts.primary ? " btn-primary btn-big" : "") + (opts.quiet ? " btn-quiet" : ""), type: "button",
  "data-k": opts.k, disabled: opts.disabled, onclick: opts.onclick }, el("span", null, label));
const foot = (...nodes) => el("div", "setup-foot", ...nodes);
const go = (step) => () => { U.step = step; U.note = ""; renderSetup(true); };
const guard = (fn) => async (ev) => {
  const b = ev.currentTarget;
  b.disabled = true;
  U.note = "";
  try { await fn(); } catch (e) { U.note = errorText({ code: e.code, message: e.message }) || e.message; b.disabled = false; }
  renderSetup(true);
};

export function renderSetup(force) {
  if (!U.open) return;
  const body = $("setup-body");
  const active = document.activeElement;
  const typing = body.contains(active) && ["INPUT", "TEXTAREA"].includes(active.tagName);
  const signature = JSON.stringify([U.step, U.info, U.pick, U.server, U.note, U.plan, S.eng && [S.eng.job, S.eng.process, S.eng.models, S.eng.model]]);
  if (!force && (signature === U.signature || typing)) return;
  const stepChanged = body.dataset.step !== U.step;
  U.signature = signature;
  const openKeys = [...body.querySelectorAll("details[open][data-k]")].map((d) => d.dataset.k);
  const focusKey = body.contains(active) && active.dataset ? active.dataset.k : null;
  body.dataset.step = U.step;
  body.textContent = "";
  const nodes = !U.info || !S.eng ? [el("div", "setup-wait", el("span", "spinner"), el("span", null, t("su_looking")))] : { where, connect, model, install, ready }[U.step]();
  body.append(...nodes.filter(Boolean));
  for (const k of openKeys) { const d = body.querySelector(`details[data-k="${k}"]`); if (d) d.open = true; }
  const again = focusKey && body.querySelector(`[data-k="${focusKey}"]`);
  if (again) again.focus();
  else if (stepChanged) { const h = body.querySelector("h1"); if (h) h.focus(); }
}

// ---- 1. where do the models run? ----------------------------------------------------------------------------------
function where() {
  const info = U.info;
  if (info.proposal === "endpoint") {
    const servers = info.endpoints;
    const chosen = servers.find((e) => e.base_url === U.server) || servers[0];
    const url = chosen.base_url.replace(/^https?:\/\//, "").replace(/\/v1$/, "");
    const out = [title(t("su_found_title", { name: chosen.name })), lede(t("su_found_text", { n: chosen.models, url }))];
    out.push(lede(isMac() ? t("su_found_mac") : t("su_found_managed")));
    if (servers.length > 1) {
      out.push(el("div", { class: "options", role: "radiogroup" }, servers.map((e) => el("label", "option" + (e.base_url === chosen.base_url ? " picked" : ""),
        el("input", { type: "radio", name: "server", checked: e.base_url === chosen.base_url, "data-k": "srv-" + e.base_url, onchange: () => { U.server = e.base_url; renderSetup(true); } }),
        el("span", "option-text", el("strong", null, e.name), el("span", "muted", e.base_url.replace(/^https?:\/\//, "") + " · " + t("conn_models_n", { n: e.models }))))))); 
    }
    if (U.note) out.push(el("p", { class: "note bad", role: "alert" }, U.note));
    out.push(foot(btn(t("su_start"), { primary: true, k: "start", onclick: guard(async () => { await useServer(chosen.name, chosen.base_url, null); await finish(); }) })));
    const others = [btn(t("su_other_server"), { quiet: true, k: "other-server", onclick: go("connect") })];
    if (info.supported) others.push(btn(t("su_other_local"), { quiet: true, k: "other-local", onclick: go("model") }));
    const busy = info.busy_gpus || [];
    out.push(el("details", { class: "more", "data-k": "other" }, el("summary", null, icon("chevron"), el("span", null, t("su_other"))),
      el("div", "more-body", el("div", "setup-links", ...others),
        info.supported && busy.length ? el("p", "note warn", t("conn_local_busy", { names: busyNames(busy) })) : null)));
    return out;
  }
  // nothing answers and this computer can run the engine
  const rec = S.eng.models.find((m) => m.key === info.recommended);
  return [title(t("su_local_title")), lede(t("su_local_text")), machineLines(info.machine),
    rec ? checkList(rec.fit.lines.filter((l) => l.level !== "ok")) : null,
    foot(btn(t("su_install_here"), { primary: true, k: "install-here", onclick: go("model") }), btn(t("su_connect_instead"), { quiet: true, k: "connect-instead", onclick: go("connect") }))];
}

// ---- a server somewhere ---------------------------------------------------------------------------------------------
function connect() {
  const info = U.info;
  const url = el("input", { placeholder: "http://127.0.0.1:8080/v1", autocomplete: "off", spellcheck: "false", "data-k": "url", "aria-label": t("conn_url") });
  const key = el("input", { type: "password", autocomplete: "off", "data-k": "key", "aria-label": t("conn_key") });
  const submit = guard(async () => {
    const address = url.value.trim();
    if (!address) { url.focus(); return; }
    await useServer("server", address, key.value || null);
    const check = await api("GET", "/api/check");
    if (!check.ok) throw new Error(t("su_connect_fail", { url: address }) + (check.error ? ` (${check.error})` : ""));
    await finish();
  });
  url.addEventListener("keydown", (e) => { if (e.key === "Enter") $("setup-body").querySelector('[data-k="connect"]').click(); });
  const out = [title(t("su_connect_title")), lede(isMac() ? t("su_connect_mac") : info.supported ? t("su_connect_any") : t("su_connect_unsupported"))];
  out.push(el("div", "form", el("label", "field", el("span", "field-label", t("conn_url")), url),
    el("label", "field", el("span", "field-label", t("conn_key"), el("span", "field-hint", t("conn_key_none"))), key)));
  if (U.note) out.push(el("p", { class: "note bad", role: "alert" }, U.note));
  const back = info.proposal !== "connect" ? btn(t("su_back"), { quiet: true, k: "back", onclick: go("where") }) : null;
  out.push(foot(btn(t("su_connect"), { primary: true, k: "connect", onclick: submit }), back));
  out.push(el("div", "info-box", el("p", null, t("conn_tunnel")), tunnel()));
  if (!isMac() && !info.supported && info.why_not.length) {
    out.push(el("details", { class: "more", "data-k": "why" }, el("summary", null, icon("chevron"), el("span", null, t("su_why"))), el("div", "more-body", checkList(info.why_not))));
  }
  return out;
}

// ---- 2. which model -------------------------------------------------------------------------------------------------
function model() {
  const e = S.eng;
  const fit = e.models.filter((m) => m.fit.ok), unfit = e.models.filter((m) => !m.fit.ok);
  const rank = (m) => (m.recommended ? 0 : m.optional ? 3 : m.installed ? 1 : 2);
  fit.sort((a, b) => rank(a) - rank(b));
  if (!fit.some((m) => m.key === U.pick)) U.pick = fit.length ? fit[0].key : "";
  const picked = fit.find((m) => m.key === U.pick);
  const out = [title(t("su_model_title")), lede(t("su_model_text"))];
  const option = (m) => el("label", "option model-option" + (m.key === U.pick ? " picked" : ""),
    el("input", { type: "radio", name: "model", checked: m.key === U.pick, "data-k": "pick-" + m.key, onchange: () => { U.pick = m.key; renderSetup(true); } }),
    modelCard(m, e, { readOnly: true }));
  // The usual choices first; special editions (Flash-Next with the vision tower) one click away.
  const main = fit.filter((m) => !m.optional || m.installed || m.key === U.pick), extra = fit.filter((m) => !main.includes(m));
  out.push(el("div", { class: "options", role: "radiogroup" }, main.map(option)));
  if (extra.length) {
    out.push(el("details", { class: "more", "data-k": "editions" }, el("summary", null, icon("chevron"), el("span", null, `${t("su_more_editions")} (${extra.length})`)),
      el("div", "more-body options", extra.map(option))));
  }
  if (unfit.length) {
    out.push(el("details", { class: "more", "data-k": "unfit" }, el("summary", null, icon("chevron"), el("span", null, `${t("m_notfit")} (${unfit.length})`)),
      el("div", "more-body cards", unfit.map((m) => modelCard(m, e, { readOnly: true })))));
  }
  if (U.note) out.push(el("p", { class: "note bad", role: "alert" }, U.note));
  // What the main button does depends on what is on disk for the chosen model. Files the app has not verified are
  // never used or changed without the user choosing: verify them, a new folder next to them, or as they are.
  const state = picked ? picked.state : "absent";
  const onDisk = ["foreign", "different", "outside"].includes(state);
  const ENGINE_BYTES = 0.95e9;
  const modelBytes = picked ? picked.size_gb * 1e9 : 0;
  const downloads = picked && !picked.installed && state !== "foreign";  // (verifying reads, it does not download)
  const withEngine = !e.release && picked && !picked.installed;
  const total = bytes((downloads ? modelBytes : 0) + (withEngine ? ENGINE_BYTES : 0));
  let label = t("su_download", { size: total });
  if (picked && picked.installed) label = t("su_use");
  else if (state === "foreign") label = t("su_verify_use");
  else if (onDisk) label = t("m_new_folder", { size: total });
  const sizes = withEngine && downloads ? el("span", "muted small setup-sizes", t("su_download_note", { model: bytes(modelBytes), engine: bytes(ENGINE_BYTES) })) : null;
  const choose = (body, install) => guard(async () => {
    await act("/api/engine/use", Object.assign({ model: picked.key }, body));
    await loadState();
    if (install || !S.eng.release) { await act("/api/engine/install", Object.assign({ model: picked.key }, install || {})); U.step = "install"; } else U.step = "ready";
  });
  const primary = picked && picked.installed ? choose({}, null) : choose({}, state === "different" || state === "outside" ? { new_folder: true } : {});
  out.push(el("div", "setup-foot sticky", btn(label, { primary: true, k: "get", disabled: !picked, onclick: primary }),
    onDisk ? btn(t("m_as_is"), { quiet: true, k: "as-is", onclick: choose({ as_is: true }, null) }) : null,
    btn(t("su_back"), { quiet: true, k: "back", onclick: U.from === "panel" ? () => finish() : go("where") }), sizes));
  return out;
}

// ---- 3. install -----------------------------------------------------------------------------------------------------
function install() {
  const e = S.eng, job = e.job, m = chosenModel();
  if (job && job.state === "done" && m && m.installed) { U.step = "ready"; return ready(); }
  if (job && job.state === "failed" && ["different_version", "outside"].includes(job.error && job.error.code)) {
    U.step = "model";  // what is on disk is not what the app can use: back to the choice, which now says so
    return model();
  }
  const out = [title(t("su_install_title")), lede(t("su_install_text"))];
  if (m) out.push(el("p", "setup-what", el("strong", null, `${m.title} · ${m.edition}`), el("span", "muted", " · " + tilde(e.dir, S.home))));
  if (job && job.state === "running") out.push(progressBlock(job));
  else if (job && job.state === "failed") {
    out.push(el("div", { class: "fail", role: "alert" }, el("strong", null, t("job_failed")), el("span", "fail-detail", errorText(job.error))));
  } else out.push(el("p", "note", t("job_cancelled")));
  if (U.note) out.push(el("p", { class: "note bad", role: "alert" }, U.note));
  const running = job && job.state === "running";
  out.push(foot(
    running ? btn(t("su_background"), { primary: true, k: "background", onclick: () => finish() })
      : btn(t(job && job.state === "failed" ? "try_again" : "continue_download"), { primary: true, k: "again", onclick: guard(() => act("/api/engine/install", { model: e.model })) }),
    running ? null : btn(t("su_background"), { quiet: true, k: "background", onclick: () => finish() })));
  return out;
}

// ---- 4. start, and tune if wanted -----------------------------------------------------------------------------------
function ready() {
  const e = S.eng, p = e.process, m = chosenModel(), u = units();
  const name = modelName(m);
  if (p.state === "running") {
    if (!U.plan || U.plan.model !== e.model) {
      U.plan = { model: e.model };
      api("GET", "/api/tune/plan?model=" + encodeURIComponent(e.model)).then((plan) => { U.plan = plan; renderSetup(true); }).catch(() => {});
    }
    const out = [title(t("su_ready_title")), lede(t("su_ready_text", { model: name }))];
    if (U.note) out.push(el("p", { class: "note bad", role: "alert" }, U.note));
    if (U.plan && U.plan.tunable) {
      out.push(el("div", "card tune-card", el("div", "card-head", icon("bolt"), el("strong", null, t("su_tune_title"))),
        el("p", "card-note", t("su_tune_text", { t: duration(U.plan.seconds, u) })),
        el("div", "card-actions", btn(t("su_tune_now"), { k: "tune", onclick: guard(async () => { await act("/api/tune/start", { long_context: false }); await finish("tuning"); }) }))));
    }
    out.push(foot(btn(t("su_start"), { primary: true, k: "chat", onclick: () => finish() })));
    return out;
  }
  if (p.state === "starting" || p.state === "stopping") {
    return [title(t("su_starting_title")), lede(t("su_starting_text", { model: name })), el("div", "bar-track indeterminate", el("span")),
      foot(btn(t("su_background"), { quiet: true, k: "background", onclick: () => finish() }))];
  }
  // it did not start: say why, and leave the details one click away
  const reason = p.error ? errorText(p.error) || p.error.message : t("st_stopped_sub", { model: name });
  return [title(t(p.state === "failed" ? "st_failed_title" : "st_stopped_title")),
    el("div", { class: p.error ? "fail" : "note", role: p.error ? "alert" : null }, el("strong", null, reason), p.error && p.error.log ? el("pre", "log", p.error.log.join("\n")) : null),
    U.note ? el("p", { class: "note bad", role: "alert" }, U.note) : null,
    foot(btn(t("start_engine"), { primary: true, k: "start-engine", onclick: guard(() => act("/api/engine/start")) }),
      btn(t("details"), { quiet: true, k: "details", onclick: () => finish("engine") }))];
}

export { loadModels };
