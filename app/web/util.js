// capy-flash app — small helpers shared by the page's modules. No framework, no build step.

export const $ = (id) => document.getElementById(id);

/** el("div", "cls", "text") or el("button", {class, type, title, onclick, ...}, child, child...) */
export function el(tag, props, ...children) {
  const e = document.createElement(tag);
  if (typeof props === "string") e.className = props;
  else if (props) {
    for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (k === "disabled" || k === "hidden" || k === "checked" || k === "open") e[k] = !!v;
      else if (k === "value") e.value = v;
      else e.setAttribute(k, v === true ? "" : v);
    }
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    e.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return e;
}

export function icon(name, cls) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "ico" + (cls ? " " + cls : ""));
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(NS, "use");
  use.setAttribute("href", "#i-" + name);
  svg.append(use);
  return svg;
}

export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const uid = () => Array.from(crypto.getRandomValues(new Uint8Array(8)), (b) => b.toString(16).padStart(2, "0")).join("");

export class ApiError extends Error {
  constructor(status, message, code) { super(message); this.status = status; this.code = code || ""; }
}

export async function api(method, path, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  let r;
  try { r = await fetch(path, opt); } catch (e) { throw new ApiError(0, e.message, "app_gone"); }
  let data = null;
  try { data = await r.json(); } catch (_) { /* empty */ }
  if (!r.ok) {
    const err = (data && data.error) || {};
    throw new ApiError(r.status, err.message || r.statusText, err.code);
  }
  return data;
}

export async function copyText(text, button, doneLabel) {
  try { await navigator.clipboard.writeText(text); } catch (_) {
    const ta = el("textarea", { class: "sr" }); ta.value = text; document.body.append(ta); ta.select();
    document.execCommand("copy"); ta.remove();
  }
  if (!button) return;
  button.classList.add("done");
  const label = button.querySelector("span") || button;
  const old = label.textContent;
  if (doneLabel) label.textContent = doneLabel;
  setTimeout(() => { button.classList.remove("done"); if (doneLabel) label.textContent = old; }, 1300);
}

// ---- numbers, as people read them -------------------------------------------------------------------------------
let LOCALE = "en";
export const setLocale = (l) => { LOCALE = l; };
export const num = (v, digits = 0) => Number(v).toLocaleString(LOCALE, { minimumFractionDigits: digits, maximumFractionDigits: digits });

export function bytes(n) {
  if (n >= 1e9) return num(n / 1e9, n >= 1e10 ? 0 : 1) + " GB";
  if (n >= 1e6) return num(n / 1e6, 0) + " MB";
  return num(Math.max(n, 0) / 1e3, 0) + " kB";
}

/** Seconds as "40 s", "12 min", "2 h 10 min". */
export function duration(seconds, units) {
  const s = Math.max(0, Math.round(seconds));
  if (s < 90) return s + " " + units.s;
  const m = Math.round(s / 60);
  if (m < 90) return m + " " + units.min;
  return Math.floor(m / 60) + " " + units.h + (m % 60 ? " " + (m % 60) + " " + units.min : "");
}

/** A path with the home folder as "~", as people write it. */
export function tilde(path, home) {
  return home && (path === home || path.startsWith(home + "/")) ? "~" + path.slice(home.length) : path;
}

export function debounce(fn, ms) {
  let timer = 0;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}
