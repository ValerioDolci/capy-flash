// capy-flash app — the page's entry point: load the state, wire the parts together, keep the status honest.
// No framework, no build step, no network beyond this app's own server.
import { $, api } from "/util.js";
import { t, applyLanguage } from "/i18n.js";
import { S, onChange, changed, loadState, loadModels, loadEngine, scheduleEngine, isLocal, status } from "/state.js";
import { initChat, renderLog, renderEmpty, renderModels, loadList, newChat, closeRail } from "/chat.js";
import { initPanel, openPanel, closePanel, render as renderPanel, setPanelHooks } from "/panel.js";
import { startSetup, renderSetup, setSetupHooks, setupOpen } from "/setup.js";

const systemLight = window.matchMedia("(prefers-color-scheme: light)");

function applyTheme() {
  const theme = S.settings ? S.settings.theme : "dark";
  document.documentElement.dataset.theme = theme === "light" || (theme === "system" && systemLight.matches) ? "light" : "dark";
}

function gate(titleKey, textKey) {
  $("shell").hidden = true;
  $("setup").hidden = true;
  closePanel();
  $("gate").hidden = false;
  $("gate-title").textContent = t(titleKey);
  $("gate-text").textContent = t(textKey);
}

/** The status card in the rail: one place that says where the answers come from. */
function renderStatus() {
  const st = status();
  $("status-dot").className = "dot " + st.tone;
  $("status-title").textContent = st.card[0];
  $("status-sub").textContent = st.card[1];
  const bar = $("status-bar");
  bar.hidden = !(st.progress >= 0);
  if (st.progress >= 0) $("status-bar-fill").style.width = Math.round(st.progress * 100) + "%";
  $("status").title = typeof st.sub === "string" ? st.sub : "";
}

let lastModels = "";
function refresh() {
  if (!S.settings) return;
  const sig = JSON.stringify([S.models.map((m) => m.id), S.eng && S.eng.model, S.settings.backend.kind]);
  if (sig !== lastModels) { lastModels = sig; renderModels(); }
  renderStatus();
  renderEmpty();
  renderPanel();
  renderSetup();
}

async function afterBackendChange() {
  lastModels = "";
  if (isLocal()) await loadEngine();
  scheduleEngine();
  changed();
}

async function quit() {
  try { await api("POST", "/api/quit", {}); } catch (_) { /* closing */ }
  gate("closed_title", "closed_text");
}

async function init() {
  try {
    await loadState();
  } catch (e) {
    applyLanguage("auto");
    gate("gate_title", "gate_text");
    return;
  }
  applyTheme();
  applyLanguage(S.settings.language);
  $("thinking").checked = S.settings.thinking !== false;
  initChat();
  initPanel();
  setPanelHooks({ startSetup, applyTheme, afterBackendChange, quit });
  setSetupHooks({ onFinish: async () => { await afterBackendChange(); await loadModels(); renderLog(); $("input").focus(); } });
  onChange(refresh);
  systemLight.addEventListener("change", applyTheme);

  $("status").addEventListener("click", () => { closeRail(); openPanel(isLocal() ? "engine" : "connection"); });
  $("open-tools").addEventListener("click", () => { closeRail(); openPanel("tools"); });
  $("open-settings").addEventListener("click", () => { closeRail(); openPanel("settings"); });
  document.addEventListener("capy:open", (e) => openPanel(e.detail));
  document.addEventListener("keydown", (e) => {
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName);
    if (e.key === "Escape" && !$("panel").open) {
      if (S.busy) S.busy.abort();
      else closeRail();
    } else if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === "o") {
      e.preventDefault();
      if (!setupOpen()) { closePanel(); newChat(); }
    } else if (e.key === "/" && !typing && !$("panel").open && !setupOpen()) {
      e.preventDefault();
      $("input").focus();
    }
  });
  // A server that was off may be back: look again when the window gets the focus, and now and then while it is down.
  let lastCheck = Date.now();
  const recheck = () => {
    if (isLocal() || S.busy || Date.now() - lastCheck < 8000) return;
    lastCheck = Date.now();
    loadModels();
  };
  window.addEventListener("focus", recheck);
  setInterval(() => { if (S.modelsError && !document.hidden) recheck(); }, 15000);

  if (isLocal()) await loadEngine();
  await Promise.all([loadModels(), loadList()]);
  renderLog();
  refresh();
  scheduleEngine();
  if (!S.settings.setup_done) {
    await startSetup();
  } else {
    $("shell").hidden = false;
    $("input").focus();
  }
}

init();
