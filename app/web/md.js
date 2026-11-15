// capy-flash app — a small Markdown renderer for model output. Safe by construction: every character of the
// source is escaped first, and only http/https links become links.
import { esc } from "/util.js";
import { t } from "/i18n.js";

function inline(text) {
  const codes = [];
  let s = esc(text).replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return "\u0000" + (codes.length - 1) + "\u0000"; });
  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
       .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>")
       .replace(/(^|[\s(])_([^_\n]+)_(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>")
       .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => "<code>" + codes[+i] + "</code>");
}

function blocks(text) {
  const lines = text.split("\n");
  const out = [];
  let i = 0;
  const isTableSep = (l) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l);
  const cells = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => inline(c.trim()));
  const isItem = (l) => /^\s*([-*+]|\d+[.)])\s+/.test(l);
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    let m;
    if ((m = line.match(/^(#{1,6})\s+(.*)$/))) { const n = Math.min(m[1].length, 4); out.push(`<h${n}>${inline(m[2])}</h${n}>`); i++; continue; }
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(line)) { out.push("<hr>"); i++; continue; }
    if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const head = cells(line); i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) { rows.push(cells(lines[i])); i++; }
      out.push('<div class="table-wrap"><table><thead><tr>' + head.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>" +
        rows.map((r) => "<tr>" + r.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("") + "</tbody></table></div>");
      continue;
    }
    if (/^\s*>/.test(line)) {
      const quote = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) { quote.push(lines[i].replace(/^\s*>\s?/, "")); i++; }
      out.push("<blockquote>" + blocks(quote.join("\n")) + "</blockquote>");
      continue;
    }
    if (isItem(line)) {
      const ordered = /^\s*\d+[.)]/.test(line);
      const items = [];
      while (i < lines.length && (isItem(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length))) {
        if (isItem(lines[i])) items.push(lines[i].replace(/^\s*([-*+]|\d+[.)])\s+/, ""));
        else items[items.length - 1] += " " + lines[i].trim();
        i++;
      }
      const tag = ordered ? "ol" : "ul";
      out.push(`<${tag}>` + items.map((it) => `<li>${inline(it)}</li>`).join("") + `</${tag}>`);
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|\s*>|\s*([-*+]|\d+[.)])\s+)/.test(lines[i]) &&
           !(lines[i].includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1]))) { para.push(lines[i]); i++; }
    out.push("<p>" + para.map(inline).join("<br>") + "</p>");
  }
  return out.join("");
}

export function markdown(src) {
  const parts = src.split(/^```/m);
  let html = "";
  parts.forEach((part, k) => {
    if (k % 2 === 0) { html += blocks(part); return; }
    const nl = part.indexOf("\n");
    const lang = (nl < 0 ? part : part.slice(0, nl)).trim().replace(/[^\w+#.-]/g, "");
    const code = nl < 0 ? "" : part.slice(nl + 1).replace(/\n$/, "");
    html += `<div class="code"><div class="code-head"><span>${esc(lang || "code")}</span>` +
      `<button class="copy" type="button"><span>${esc(t("copy"))}</span></button></div><pre><code>${esc(code)}</code></pre></div>`;
  });
  return html;
}
