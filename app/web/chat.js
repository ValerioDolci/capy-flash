// capy-flash app — the chat: conversations, messages, streaming, the speed strip.
import { $, el, icon, uid, api, copyText, num } from "/util.js";
import { t } from "/i18n.js";
import { markdown } from "/md.js";
import { S, status, chatModels, isLocal, loadModels, act, chosenModel, changed, visionOf } from "/state.js";

export const go = (section) => document.dispatchEvent(new CustomEvent("capy:open", { detail: section }));

function newConversation() {
  return { id: uid(), title: "", model: $("model").value || (S.settings && S.settings.model) || "", system: "", messages: [] };
}

// ---- the speed strip: the app's signature ------------------------------------------------------------------------
function sparkline(values) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 120 20");
  svg.setAttribute("aria-hidden", "true");
  const max = Math.max(...values, 1);
  const step = 120 / Math.max(values.length - 1, 1);
  const pts = values.map((v, i) => [i * step, 19 - (v / max) * 17]);
  const line = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  const area = document.createElementNS(NS, "path");
  area.setAttribute("d", line + ` L ${pts[pts.length - 1][0].toFixed(1)} 20 L 0 20 Z`);
  area.setAttribute("class", "area");
  const path = document.createElementNS(NS, "path");
  path.setAttribute("d", line);
  path.setAttribute("class", "line");
  svg.append(area, path);
  return svg;
}

function speedStrip(m) {
  const wrap = el("div", "speed");
  const stats = m.stats;
  if (stats) {
    if (stats.timeline && stats.timeline.length > 1) wrap.append(sparkline(stats.timeline));
    const part = (...nodes) => wrap.append(el("span", "speed-part", ...nodes));
    if (stats.ttft != null) part(t("ttft") + " ", el("b", null, num(stats.ttft, 2) + " s"));
    if (stats.tps != null) part(el("b", null, num(stats.tps, 1)), " " + t("tok_s"));
    if (stats.tokens != null) part(num(stats.tokens) + " " + t("tokens"));
    if (stats.stopped) part(t("stopped"));
    if (stats.prompt_tps) wrap.title = `prefill ${num(stats.prompt_n || 0)} tokens · ${num(stats.prompt_tps)} tok/s` + (stats.cache_n ? ` · cache ${num(stats.cache_n)}` : "");
  }
  if (m.model) wrap.append(el("span", "speed-part speed-model", m.model));
  return wrap;
}

// ---- messages ---------------------------------------------------------------------------------------------------
function messageNode(m, isLast) {
  if (m.role === "user") {
    const node = el("div", "msg msg-user");
    const bubble = el("div", "bubble");
    for (const src of m.images || []) bubble.append(el("img", { src, alt: "" }));
    bubble.append(document.createTextNode(m.content || ""));
    node.append(bubble);
    return node;
  }
  const node = el("div", "msg msg-assistant" + (m.error ? " error" : ""));
  if (m.reasoning) {
    node.append(el("details", "think", el("summary", null, icon("chevron"), t("reasoning")), el("div", "think-body", m.reasoning)));
  }
  if (m.error) {
    node.append(el("div", { class: "fail", role: "alert" }, el("strong", null, t("err_title")), el("span", "fail-detail", m.error),
      el("button", { class: "btn btn-small", type: "button", onclick: regenerate }, icon("retry"), el("span", null, t("retry")))));
    return node;
  }
  const content = el("div", "content");
  content.innerHTML = markdown(m.content || "");
  if (m.images_dropped) content.append(el("p", "dropped", t("images_dropped", { model: m.model || "" })));
  const foot = el("div", "msg-foot", speedStrip(m));
  const actions = el("div", "msg-actions");
  const copy = el("button", { class: "ghost-btn", type: "button", title: t("copy"), "aria-label": t("copy") }, icon("copy"), el("span", "sr", t("copy")));
  copy.addEventListener("click", () => copyText(m.content || "", copy));
  actions.append(copy);
  if (isLast) {
    actions.append(el("button", { class: "ghost-btn", type: "button", title: t("regenerate"), "aria-label": t("regenerate"), onclick: regenerate }, icon("retry")));
  }
  foot.append(actions);
  node.append(content, foot);
  return node;
}

export function renderLog() {
  const log = $("log");
  log.querySelectorAll(".msg, .resend").forEach((n) => n.remove());
  const msgs = S.conv ? S.conv.messages : [];
  $("empty").hidden = msgs.length > 0;
  msgs.forEach((m, i) => log.append(messageNode(m, i === msgs.length - 1)));
  if (msgs.length && msgs[msgs.length - 1].role === "user" && !S.busy) {  // a question left without an answer
    log.append(el("div", "resend", el("button", { class: "btn btn-small", type: "button", onclick: regenerate }, icon("retry"), el("span", null, t("regenerate")))));
  }
  log.scrollTop = log.scrollHeight;
  renderEmpty();
}

/** The empty chat says where the answers come from, or what is missing and how to fix it. */
export function renderEmpty() {
  if (!S.settings) return;
  const st = status();
  const model = $("model").value || "";
  $("empty-title").textContent = st.title;
  $("empty-sub").textContent = typeof st.sub === "function" ? st.sub(isLocal() ? "" : model) : st.sub;
  const bar = $("empty-progress");
  bar.hidden = st.progress == null;
  bar.classList.toggle("indeterminate", st.progress === -1);
  if (st.progress != null && st.progress >= 0) $("empty-progress-fill").style.width = Math.round(st.progress * 100) + "%";
  const actions = $("empty-actions");
  const want = (st.actions || []).join(",");
  if (actions.dataset.for !== st.kind + ":" + want) {  // rebuilt only when the state changes: buttons keep focus
    actions.dataset.for = st.kind + ":" + want;
    actions.textContent = "";
    const note = el("p", { class: "note bad", role: "alert" });
    const button = (label, primary, fn) => el("button", { class: "btn" + (primary ? " btn-primary" : ""), type: "button", onclick: fn }, label);
    const run = (path, body) => async (ev) => {
      ev.currentTarget.disabled = true;
      try { await act(path, body); } catch (e) { note.textContent = e.message; }
    };
    (st.actions || []).forEach((a, i) => {
      if (a === "retry") actions.append(button(t("try_again"), i === 0, () => loadModels()));
      else if (a === "connection") actions.append(button(t("connection"), i === 0, () => go("connection")));
      else if (a === "start") actions.append(button(t("start_engine"), i === 0, run("/api/engine/start")));
      else if (a === "install") actions.append(button(t("continue_download"), i === 0, run("/api/engine/install", { model: S.eng.model })));
      else if (a === "engine" || a === "tuning") actions.append(button(t("details"), false, () => go(a)));
      else if (a === "models") actions.append(button(st.kind === "installing" ? t("details") : t("choose_model"), st.kind !== "installing", () => go("models")));
    });
    actions.append(note);
  }
  const can = st.can && !!model;
  updateAttach();
  $("send").disabled = !can;
  $("input").placeholder = can || st.kind === "checking" ? t("input_placeholder") : st.title;
  $("composer-box").classList.toggle("off", !can && st.kind !== "checking");
}

// ---- conversations ----------------------------------------------------------------------------------------------
function dayGroup(ts) {
  const day = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const diff = (day(new Date()) - day(new Date(ts * 1000))) / 86400000;
  return diff <= 0 ? "today" : diff === 1 ? "yesterday" : "earlier";
}

let renaming = "";  // the conversation whose title is being edited (double click or F2)

export function renderList() {
  const nav = $("conv-list");
  const typed = nav.querySelector(".conv-rename");
  const draft = typed ? typed.value : null;  // a list refresh must not eat what is being typed
  nav.textContent = "";
  let group = "";
  for (const c of S.list) {
    const g = dayGroup(c.updated);
    if (g !== group) { group = g; nav.append(el("div", "conv-group", t(g))); }
    const active = S.conv && S.conv.id === c.id;
    if (renaming === c.id) {
      nav.append(el("div", "conv" + (active ? " active" : ""), renameInput(c, draft)));
      continue;
    }
    const open = el("button", { class: "conv-open", type: "button", title: t("rename"), "aria-current": active ? "true" : null }, c.title || t("untitled"));
    open.addEventListener("click", () => openConversation(c.id));
    open.addEventListener("dblclick", () => { renaming = c.id; renderList(); });
    open.addEventListener("keydown", (e) => { if (e.key === "F2") { renaming = c.id; renderList(); } });
    const del = el("button", { class: "conv-del", type: "button", title: t("delete"), "aria-label": t("delete") }, icon("trash"));
    let armed = 0;
    del.addEventListener("click", async () => {
      if (!armed) {  // first click asks, in place; the second one deletes
        del.classList.add("armed");
        del.textContent = t("delete_confirm");
        armed = setTimeout(() => { armed = 0; del.classList.remove("armed"); del.textContent = ""; del.append(icon("trash")); }, 3000);
        return;
      }
      clearTimeout(armed);
      await api("DELETE", "/api/conversations/" + c.id, {});
      if (S.conv && S.conv.id === c.id) { S.conv = newConversation(); renderLog(); }
      await loadList();
    });
    nav.append(el("div", "conv" + (active ? " active" : ""), open, del));
  }
  const input = nav.querySelector(".conv-rename");
  if (input && document.activeElement !== input) { input.focus(); if (draft == null) input.select(); }
}

function renameInput(c, draft) {
  const input = el("input", { class: "conv-rename", value: draft != null ? draft : c.title || "", "aria-label": t("rename") });
  const finish = async (save) => {
    if (renaming !== c.id || !input.isConnected) return;  // replaced by a refresh of the list: not the user leaving
    renaming = "";
    const title = input.value.trim().slice(0, 80);
    if (save && title && title !== c.title) {
      const conv = await api("GET", "/api/conversations/" + c.id);
      conv.title = title;
      await api("PUT", "/api/conversations/" + c.id, conv);
      if (S.conv && S.conv.id === c.id) S.conv.title = title;
    }
    await loadList();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    if (e.key === "Escape") { e.stopPropagation(); finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
  return input;
}

export async function loadList() { S.list = (await api("GET", "/api/conversations")).conversations; renderList(); }

async function openConversation(id) {
  if (S.busy) return;
  S.conv = await api("GET", "/api/conversations/" + id);
  if (S.conv.model && [...$("model").options].some((o) => o.value === S.conv.model)) $("model").value = S.conv.model;
  $("system").value = S.conv.system || "";
  showSystem(!!S.conv.system);
  renderLog();
  renderList();
  closeRail();
}

async function saveConversation() {
  if (!S.conv) return;
  const kept = S.conv.messages.filter((m) => !m.error);  // a failed attempt is not part of the conversation
  if (!kept.length) return;
  if (!S.conv.title) {
    const first = kept.find((m) => m.role === "user");
    S.conv.title = ((first && first.content) || "").replace(/\s+/g, " ").trim().slice(0, 60) || t("untitled");
  }
  await api("PUT", "/api/conversations/" + S.conv.id, Object.assign({}, S.conv, { messages: kept }));
  await loadList();
}

export function newChat() {
  if (S.busy) return;
  S.conv = newConversation();
  $("system").value = "";
  showSystem(false);
  renderLog();
  renderList();
  closeRail();
  $("input").focus();
}

// ---- the model picker ---------------------------------------------------------------------------------------------
export function renderModels() {
  const select = $("model");
  const keep = select.value || (S.conv && S.conv.model) || S.settings.model;
  select.textContent = "";
  const chat = chatModels();
  const local = isLocal() ? chosenModel() : null;
  for (const m of chat) {
    // On this computer the engine serves one model: show its name rather than the API id.
    const label = local && chat.length === 1 ? `${local.title} · ${local.edition}` : m.id;
    select.append(el("option", { value: m.id, title: (m.name || "") + (m.description ? " — " + m.description : "") }, label));
  }
  if (!chat.length) select.append(el("option", { value: "" }, "—"));
  select.parentElement.hidden = !chat.length;  // nothing to pick: the empty chat says why
  select.classList.toggle("single", chat.length === 1);
  fitModelPicker();
  if (keep && chat.some((m) => m.id === keep)) select.value = keep;
  select.disabled = !chat.length || !!S.busy;
}

/** A native select is as wide as its longest option: size it to the chosen one, so the arrow sits next to the name.
 *  Browsers with `field-sizing: content` do it themselves (app.css); this is for the others. */
function fitModelPicker() {
  const select = $("model");
  const option = select.selectedOptions[0];
  if (!option || (window.CSS && CSS.supports("field-sizing", "content"))) return;
  const style = getComputedStyle(select);
  const ctx = fitModelPicker.canvas || (fitModelPicker.canvas = document.createElement("canvas").getContext("2d"));
  ctx.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;  // measured without layout: works while hidden
  const text = Math.ceil(ctx.measureText(option.textContent).width);
  select.style.width = text ? text + (select.classList.contains("single") ? 40 : 60) + "px" : "";
}

// ---- sending ------------------------------------------------------------------------------------------------------
/** The conversation as the API wants it. Images stay in the conversation and on screen, but are sent only to a
 *  model that reads them: one image in the history would make a text-only model refuse every later turn. */
function toApiMessages(messages, vision) {
  const out = [];
  const system = $("system").value.trim();
  if (system) out.push({ role: "system", content: system });
  for (const m of messages) {
    if (m.error) continue;
    if (m.role === "user" && m.images && m.images.length && vision !== false) {
      out.push({ role: "user", content: [{ type: "text", text: m.content || "" },
        ...m.images.map((url) => ({ type: "image_url", image_url: { url } }))] });
    } else {
      out.push({ role: m.role, content: m.content || "" });
    }
  }
  return out;
}

export function send() {
  const text = $("input").value.trim();
  if (S.busy || $("send").disabled || (!text && !S.attachments.length)) return;
  if (!$("model").value) return;
  if (!S.conv) S.conv = newConversation();
  S.conv.messages.push({ role: "user", content: text, images: S.attachments.splice(0) });
  renderAttachments();
  $("input").value = "";
  autosize();
  return answer();
}

/** Answer the last user message again (after an error, a stop, or because the user asked). */
export function regenerate() {
  if (S.busy || !S.conv || $("send").disabled) return;
  const msgs = S.conv.messages;
  while (msgs.length && msgs[msgs.length - 1].role !== "user") msgs.pop();
  if (!msgs.length) return;
  return answer();
}

async function answer() {
  const model = $("model").value;
  S.conv.model = model;
  S.conv.system = $("system").value.trim();
  const reply = { role: "assistant", model, content: "", reasoning: "" };
  const vision = visionOf(model);
  const history = toApiMessages(S.conv.messages, vision);
  if (vision === false && S.conv.messages.some((m) => m.role === "user" && m.images && m.images.length)) reply.images_dropped = true;
  S.conv.messages.push(reply);
  const controller = new AbortController();
  S.busy = controller;
  setBusy(true);
  renderLog();

  const node = $("log").querySelector(".msg-assistant:last-of-type");
  const content = node.querySelector(".content");
  const strip = node.querySelector(".speed");
  node.querySelector(".msg-actions").hidden = true;
  const waiting = el("div", "waiting", el("span", "dots", el("i"), el("i"), el("i")), el("span", "wait-text"));
  content.replaceChildren(waiting);
  $("live").textContent = t("live_answering");
  let thinkNode = null;

  const body = { model, messages: history, stream: true, stream_options: { include_usage: true },
    chat_template_kwargs: { enable_thinking: $("thinking").checked } };
  if (S.settings.temperature != null && S.settings.temperature !== "") body.temperature = Number(S.settings.temperature);
  if (S.settings.max_tokens) body.max_tokens = Number(S.settings.max_tokens);

  const t0 = performance.now();
  let tFirst = null, deltas = 0, usage = null, timings = null, stopped = false;
  // Before the first token: say how long we have been waiting (a model load or a queue can take tens of seconds).
  const waitTimer = setInterval(() => {
    const s = (performance.now() - t0) / 1000;
    if (s >= 2) waiting.querySelector(".wait-text").textContent = t(s >= 8 ? "waiting_long" : "waiting", { s: s.toFixed(0) });
  }, 500);
  const buckets = [];
  let frame = 0;
  const paint = () => {
    frame = 0;
    if (reply.reasoning) {
      if (!thinkNode) {
        thinkNode = el("details", "think live", el("summary", null, icon("chevron"), t("reasoning_live")), el("div", "think-body"));
        node.insertBefore(thinkNode, content);
      }
      thinkNode.querySelector(".think-body").textContent = reply.reasoning;
    }
    if (reply.content) content.innerHTML = markdown(reply.content) + '<span class="cursor"></span>';
    const log = $("log");
    if (log.scrollHeight - log.scrollTop - log.clientHeight < 160) log.scrollTop = log.scrollHeight;
  };
  const schedule = () => { if (!frame) frame = requestAnimationFrame(paint); };

  try {
    const r = await fetch("/v1/chat/completions", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: controller.signal });
    if (!r.ok) {
      let msg = r.statusText;
      try { const j = await r.json(); msg = (j.error && (j.error.message || j.error)) || msg; } catch (_) { /* not JSON */ }
      throw new Error(`HTTP ${r.status}: ${typeof msg === "string" ? msg : JSON.stringify(msg)}`);
    }
    if (r.headers.get("X-Capy-Images") === "dropped") {  // the server found out this model has no vision
      reply.images_dropped = true;
      const known = S.models.find((x) => x.id === model);
      if (known) known.vision = false;
      updateAttach();
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const event = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        for (const line of event.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") continue;
          let j;
          try { j = JSON.parse(data); } catch (_) { continue; }
          if (j.error) throw new Error(j.error.message || JSON.stringify(j.error));
          if (j.usage) usage = j.usage;
          if (j.timings) timings = j.timings;
          const d = j.choices && j.choices[0] && j.choices[0].delta;
          if (!d) continue;
          const r1 = d.reasoning_content || d.reasoning || "";
          const c1 = d.content || "";
          if (!r1 && !c1) continue;
          const now = performance.now();
          if (tFirst === null) { tFirst = now; clearInterval(waitTimer); }
          deltas++;
          const b = Math.floor((now - tFirst) / 250);
          buckets[b] = (buckets[b] || 0) + 1;
          reply.reasoning += r1;
          reply.content += c1;
          schedule();
        }
      }
    }
  } catch (e) {
    if (e.name === "AbortError") stopped = true;
    else reply.error = e.message;
  } finally {
    clearInterval(waitTimer);
    if (frame) cancelAnimationFrame(frame);
    const tEnd = performance.now();
    const tokens = (usage && usage.completion_tokens) || deltas || 0;
    const genSeconds = tFirst !== null ? (tEnd - tFirst) / 1000 : 0;
    const clientRate = tokens > 1 && genSeconds > 0.05 ? (tokens - 1) / genSeconds : null;
    reply.stats = {
      ttft: tFirst !== null ? (tFirst - t0) / 1000 : null,
      // capy-flash reports its own decode rate (llama.cpp-style timings): the engine's number, not the network's
      tps: timings && timings.predicted_per_second > 0 ? timings.predicted_per_second : clientRate,
      tokens: tokens || null, timeline: Array.from(buckets, (v) => (v || 0) * 4), stopped,
      prompt_tps: timings && timings.prompt_per_second, prompt_n: timings && timings.prompt_n, cache_n: timings && timings.cache_n,
    };
    if (stopped && !reply.content && !reply.reasoning) S.conv.messages.pop();  // stopped before anything arrived
    S.busy = null;
    setBusy(false);
    renderLog();
    $("live").textContent = reply.error ? t("err_title") : t("live_done", { n: tokens });
    await saveConversation().catch(() => {});
    if (reply.error && isLocal()) changed();
  }
}

export function setBusy(on) {
  $("send").hidden = on;
  $("stop").hidden = !on;
  $("model").disabled = on || !chatModels().length;
  if (!on) $("input").focus();
}

// ---- composer -----------------------------------------------------------------------------------------------------
export function autosize() { const ta = $("input"); ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 240) + "px"; }

/** No attachments for a model that does not read images; say why instead of failing later. */
function updateAttach() {
  const model = $("model").value;
  const blind = !!model && visionOf(model) === false;
  // Not `disabled`: the button stays reachable, and pressing it says why there is nothing to attach.
  $("attach").setAttribute("aria-disabled", blind ? "true" : "false");
  $("attach").title = blind ? t("no_vision", { model }) : t("attach");
  if (blind && S.attachments.length) {
    S.attachments.length = 0;
    renderAttachments();
    composerNote(t("no_vision_paste", { model }));
  }
}

let noteTimer = 0;
function composerNote(text) {
  const note = $("composer-note");
  note.textContent = text;
  note.hidden = !text;
  clearTimeout(noteTimer);
  if (text) noteTimer = setTimeout(() => { note.hidden = true; }, 7000);
}

export function addImage(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const model = $("model").value;
  if (model && visionOf(model) === false) { composerNote(t("no_vision_paste", { model })); return; }
  const reader = new FileReader();
  reader.onload = () => { S.attachments.push(reader.result); renderAttachments(); };
  reader.readAsDataURL(file);
}

function renderAttachments() {
  const box = $("attachments");
  box.textContent = "";
  S.attachments.forEach((src, i) => {
    box.append(el("div", "thumb", el("img", { src, alt: "" }),
      el("button", { type: "button", "aria-label": t("delete"), onclick: () => { S.attachments.splice(i, 1); renderAttachments(); } }, icon("close"))));
  });
}

export function showSystem(open) {
  $("system-box").hidden = !open;
  $("toggle-system").setAttribute("aria-expanded", open ? "true" : "false");
  $("system-mark").hidden = !$("system").value.trim();
}

export function closeRail() { $("rail").classList.remove("open"); $("rail-scrim").hidden = true; }

export function initChat() {
  S.conv = newConversation();
  $("new-chat").addEventListener("click", newChat);
  $("send").addEventListener("click", send);
  $("stop").addEventListener("click", () => S.busy && S.busy.abort());
  $("input").addEventListener("input", autosize);
  $("input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
  });
  $("input").addEventListener("paste", (e) => {
    for (const item of e.clipboardData ? e.clipboardData.items : []) if (item.kind === "file") addImage(item.getAsFile());
  });
  $("attach").addEventListener("click", () => {
    const model = $("model").value;
    if (model && visionOf(model) === false) composerNote(t("no_vision", { model }) + ".");
    else $("file").click();
  });
  $("file").addEventListener("change", (e) => { for (const f of e.target.files) addImage(f); e.target.value = ""; });
  $("toggle-system").addEventListener("click", () => { showSystem($("system-box").hidden); if (!$("system-box").hidden) $("system").focus(); });
  $("system").addEventListener("input", () => { $("system-mark").hidden = !$("system").value.trim(); });
  $("thinking").addEventListener("change", () => api("POST", "/api/settings", { thinking: $("thinking").checked }).then((s) => { S.settings = s; }).catch(() => {}));
  $("model").addEventListener("change", () => {
    fitModelPicker();
    renderEmpty();
    api("POST", "/api/settings", { model: $("model").value }).then((s) => { S.settings = s; }).catch(() => {});
  });
  $("open-rail").addEventListener("click", () => { $("rail").classList.add("open"); $("rail-scrim").hidden = false; });
  $("rail-scrim").addEventListener("click", closeRail);
  $("log").addEventListener("click", (e) => {
    const b = e.target.closest(".copy");
    if (b && b.closest(".code")) copyText(b.closest(".code").querySelector("code").textContent, b, t("copied"));
  });
}
