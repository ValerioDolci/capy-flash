/* capy-flash v0.16.6 measurement guide. Vanilla JS, no dependencies.
   Every number on the page is read from data.json (fetched; under file:// the copy embedded in index.html is used). */
(() => {
  'use strict';

  const doc = document;
  const NS = 'http://www.w3.org/2000/svg';
  const NB = ' ';
  doc.documentElement.classList.add('js');

  let D = null;
  const DEF = { part: 'gpu', m2: '27b-nvfp4', g2: '2', m3: '27b-nvfp4', m4: '27b-nvfp4', p4: 'prose', n4: 0, x4: 'clients', L5: 0, m5: '27b-nvfp4' };
  const S = Object.assign({}, DEF);
  let VIEW = null, AT = null, AUDIT = false;

  const RES = { gpu: 'var(--gpu)', ram: 'var(--ram)', pcie: 'var(--pcie)' };
  const RESNAME = { gpu: 'GPU memory', ram: 'RAM, CPU', pcie: 'PCIe, host link' };
  const VARIES = 'repeating-linear-gradient(135deg, var(--w4) 0 5px, var(--w1) 5px 9px)';
  const wcol = (c) => (c === 0 ? VARIES : c <= 1 ? 'var(--w1)' : c === 2 ? 'var(--w2)' : c <= 4 ? 'var(--w4)' : 'var(--w8)');
  // each draft window also gets its own line style and marker shape (by columns per request)
  const wstyle = (c) => (c <= 1 ? { marker: 'square', dash: null } : c === 2 ? { marker: 'circle', dash: '6 4' }
    : c <= 4 ? { marker: 'triangle', dash: '9 3 2 3' } : { marker: 'diamond', dash: null });
  function markerEl(kind, cx, cy, r, fill, stroke) {
    const st = { fill, stroke: stroke || 'var(--panel)', 'stroke-width': 2 };
    if (kind === 'square') return s('rect', Object.assign({ x: cx - r * 0.9, y: cy - r * 0.9, width: 1.8 * r, height: 1.8 * r, rx: 1 }, st));
    if (kind === 'triangle') return s('path', Object.assign({ d: `M${cx} ${cy - r * 1.2}L${cx + r * 1.1} ${cy + r * 0.8}L${cx - r * 1.1} ${cy + r * 0.8}Z` }, st));
    if (kind === 'diamond') return s('path', Object.assign({ d: `M${cx} ${cy - r * 1.25}L${cx + r * 1.1} ${cy}L${cx} ${cy + r * 1.25}L${cx - r * 1.1} ${cy}Z` }, st));
    return s('circle', Object.assign({ cx, cy, r }, st));
  }
  function glyph(c, color) {   // legend sample: a short line in the window's style with its marker
    const g = wstyle(c);
    return s('svg', { class: 'glyph', width: 30, height: 14, viewBox: '0 0 30 14', 'aria-hidden': 'true' },
      s('line', { x1: 1, x2: 29, y1: 7, y2: 7, stroke: color, 'stroke-width': 2, 'stroke-dasharray': g.dash }),
      markerEl(g.marker, 15, 7, 4, color, 'var(--panel)'));
  }

  // ------------------------------------------------------------------ DOM helpers
  function h(tag, attrs, ...kids) {
    const el = doc.createElement(tag);
    setAttrs(el, attrs);
    append(el, kids);
    return el;
  }
  function s(tag, attrs, ...kids) {
    const el = doc.createElementNS(NS, tag);
    setAttrs(el, attrs);
    append(el, kids);
    return el;
  }
  function setAttrs(el, attrs) {
    if (!attrs) return;
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.setAttribute('class', v);
      else if (k === 'text') el.textContent = v;
      else if (k === 'style') el.style.cssText = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? '' : v);
    }
  }
  function append(el, kids) {
    for (const k of kids.flat(Infinity)) {
      if (k == null || k === false) continue;
      el.append(k instanceof Node ? k : String(k));
    }
  }
  const $ = (q, r = doc) => r.querySelector(q);
  const $$ = (q, r = doc) => Array.from(r.querySelectorAll(q));

  // ------------------------------------------------------------------ number formatting (report style: 1,354.2)
  function decs(x) { const t = String(x); const i = t.indexOf('.'); return i < 0 ? 0 : t.length - i - 1; }
  function num(x, d) {
    if (typeof x !== 'number') return String(x);
    if (d == null) d = decs(x);
    return x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  const dAt = (o, i) => (Array.isArray(o.d) ? o.d[i] : o.d);
  function fv(o, i) {
    if (o == null) return '';
    if (i != null) return num(o.v[i], dAt(o, i));
    if (Array.isArray(o.v)) return o.v.map((x, j) => num(x, dAt(o, j))).join(o.range ? '-' : ' / ');
    return num(o.v, o.d);
  }
  function vtext(o, i) { const t = fv(o, i); return o && o.u ? (o.u === '%' ? t + NB + '%' : t + NB + o.u) : t; }
  function cmark() { return h('span', { class: 'cm', title: 'C: computed in the report from measured inputs' }, 'C'); }
  function vnode(o, i, opt = {}) {
    const f = doc.createDocumentFragment();
    f.append(fv(o, i));
    if (o.u && !opt.nou) f.append(h('small', null, o.u === '%' ? '%' : o.u));
    if (o.c && !opt.noc) f.append(cmark());
    return f;
  }
  const pick = (o, i) => ({ v: o.v[i], d: dAt(o, i), u: o.u });
  function get(path) { return path.split('.').reduce((o, k) => (o == null ? o : o[k]), D); }
  const plural = (n, a, b) => (n === 1 ? a : b);

  // ------------------------------------------------------------------ controls
  let uid = 0;
  function chips(legend, opts, cur, onch, extraCls) {
    const name = 'g' + (++uid);
    const box = h('div', { class: 'chips' });
    const inputs = new Map();
    for (const o of opts) {
      const inp = h('input', { type: 'radio', name, value: String(o.value), 'aria-label': o.aria || null });
      inp.checked = String(o.value) === String(cur);
      inp.addEventListener('change', () => { if (inp.checked) onch(o.value); });
      inputs.set(String(o.value), inp);
      box.append(h('label', { class: 'chip' }, inp, h('span', null, o.label)));
    }
    const el = h('fieldset', { class: 'ctl' + (extraCls ? ' ' + extraCls : '') }, h('legend', null, legend), box);
    return {
      el,
      set(v) { for (const [k, i] of inputs) i.checked = k === String(v); },
      disable(v, dis, why) {
        const i = inputs.get(String(v));
        if (!i) return;
        i.disabled = dis;
        i.parentNode.title = dis ? why || '' : '';
      },
    };
  }
  function slider(label, n, cur, fmtOut, ticks, onch) {
    const out = h('output', null);
    const inp = h('input', { type: 'range', min: 0, max: n - 1, step: 1, value: cur, 'aria-label': label });
    const upd = () => { out.replaceChildren(...[].concat(fmtOut(+inp.value))); inp.setAttribute('aria-valuetext', out.textContent.replace('≈', 'about')); };
    inp.addEventListener('input', () => { upd(); onch(+inp.value); });
    upd();
    const el = h('div', { class: 'ctl slider' }, h('span', { class: 'ctl-l' }, label),
      h('div', { class: 'slider-row' },
        h('div', { class: 'slider-track' }, inp,
          h('div', { class: 'ticks-row', 'aria-hidden': 'true' }, ticks.map((t, i) => h('span', { style: `left:calc(8px + (100% - 16px) * ${i / (ticks.length - 1)})` }, t)))),
        out));
    return { el, set(v) { inp.value = v; upd(); } };
  }

  // ------------------------------------------------------------------ tooltip
  const tip = () => $('#tip');
  function showTip(e, text) {
    const t = tip();
    if (!t) return;
    const [a, ...rest] = text.split('\n');
    t.replaceChildren(h('b', null, a), ...rest.map((r) => h('div', null, r)));
    t.hidden = false;
    const r = t.getBoundingClientRect();
    let x = e.clientX + 14, y = e.clientY + 14;
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
    t.style.left = x + 'px';
    t.style.top = y + 'px';
  }
  function hideTip() { const t = tip(); if (t) t.hidden = true; }
  doc.addEventListener('pointerover', (e) => {
    const el = e.target.closest && e.target.closest('[data-tip]');
    if (el) showTip(e, el.getAttribute('data-tip'));
  });
  doc.addEventListener('pointermove', (e) => {
    const el = e.target.closest && e.target.closest('[data-tip]');
    if (el) showTip(e, el.getAttribute('data-tip')); else hideTip();
  });
  doc.addEventListener('pointerleave', hideTip);
  doc.addEventListener('scroll', hideTip, { passive: true });

  // ------------------------------------------------------------------ hash state
  function readHash() {
    const hs = decodeURIComponent(location.hash.slice(1));
    if (!hs.includes('=')) return;
    const p = new URLSearchParams(hs);
    for (const k of Object.keys(DEF)) {
      if (p.has(k)) S[k] = typeof DEF[k] === 'number' ? (parseInt(p.get(k), 10) || 0) : p.get(k);
    }
    VIEW = p.get('view');
    AT = p.get('at');
    AUDIT = p.get('audit') === '1';
  }
  function writeHash() {
    if (AUDIT) return;
    const p = new URLSearchParams();
    for (const k of Object.keys(DEF)) if (String(S[k]) !== String(DEF[k])) p.set(k, S[k]);
    if (VIEW) p.set('view', VIEW);
    const str = p.toString();
    try { history.replaceState(null, '', str ? '#' + str : location.pathname + location.search); }
    catch (e) { try { location.replace('#' + str); } catch (e2) { /* ignore */ } }
  }

  // keep keyboard focus across a re-render (elements carry data-fkey)
  function keepFocus(fn) {
    const a = doc.activeElement;
    const key = a && a.getAttribute && a.getAttribute('data-fkey');
    fn();
    if (key) { const el = $(`[data-fkey="${key}"]`); if (el) el.focus({ preventScroll: true }); }
  }

  // ------------------------------------------------------------------ tiny chart helper (inline SVG)
  function niceStep(span, n) {
    const raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
  }
  function yTicks(max, n = 4) {
    const st = niceStep(max, n), out = [];
    for (let v = 0; v <= max + 1e-9; v += st) out.push(v);
    if (out[out.length - 1] < max) out.push(out[out.length - 1] + st);
    return out;
  }
  const tickLabel = (v) => (v >= 1000 ? (v % 1000 === 0 ? v / 1000 + 'k' : (v / 1000).toFixed(1) + 'k') : String(v));

  function lineChart(o) {
    const W = Math.max(280, o.w), H = o.h || 250;
    const m = { l: 44, r: 16, t: 14, b: o.xlabel ? 42 : 26 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const [x0, x1] = o.xdom, yt = o.yticks, y1 = yt[yt.length - 1];
    const X = o.xlog
      ? (v) => m.l + ((Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0))) * iw
      : (v) => m.l + ((v - x0) / (x1 - x0)) * iw;
    const Y = (v) => m.t + ih - (v / y1) * ih;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': o.aria || '' });
    if (o.shadeFrom != null) {
      const xs = X(o.shadeFrom);
      svg.append(s('rect', { class: 'shade', x: xs, y: m.t, width: m.l + iw - xs, height: ih }));
      if (o.shadeLabel) svg.append(s('text', { class: 'lbl2', x: m.l + iw - 4, y: m.t + 14, 'text-anchor': 'end' }, o.shadeLabel));
    }
    const g = s('g', { class: 'grid' });
    for (const v of yt) {
      g.append(s('line', { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v) }));
      svg.append(s('text', { class: 'tick', x: m.l - 7, y: Y(v) + 4, 'text-anchor': 'end' }, tickLabel(v)));
    }
    svg.prepend(g);
    svg.append(s('line', { class: 'base', x1: m.l, x2: m.l + iw, y1: Y(0), y2: Y(0) }));
    for (const t of o.xticks) svg.append(s('text', { class: 'tick', x: X(t.v), y: m.t + ih + 16, 'text-anchor': 'middle' }, t.l));
    if (o.xlabel) svg.append(s('text', { class: 'axl', x: m.l + iw / 2, y: H - 6, 'text-anchor': 'middle' }, o.xlabel));
    if (o.cursorX != null) svg.append(s('line', { class: 'cursor', x1: X(o.cursorX), x2: X(o.cursorX), y1: m.t - 4, y2: m.t + ih }));
    const order = o.series.slice().sort((a, b) => (a.emph ? 1 : 0) - (b.emph ? 1 : 0));
    for (const se of order) {
      const pts = se.pts.filter((p) => p.y != null);
      if (!pts.length) continue;
      const d = pts.map((p, i) => (i ? 'L' : 'M') + X(p.x).toFixed(1) + ' ' + Y(p.y).toFixed(1)).join(' ');
      svg.append(s('path', { class: 'ser', d, stroke: se.color, 'stroke-width': se.width || 2, 'stroke-dasharray': se.dash || null }));
      for (const p of pts) {
        const cx = X(p.x), cy = Y(p.y), r = p.big ? 5.5 : se.r != null ? se.r : 4;
        if (r > 0) svg.append(markerEl(se.marker || 'circle', cx, cy, r, se.color));
        if (p.tip) svg.append(s('circle', { class: 'hit', cx, cy, r: 12, 'data-tip': p.tip }));
      }
    }
    const rings = (o.rings || []).map((r) => ({ cx: X(r.x), cy: Y(r.y), label: r.label }));
    for (const r of rings) svg.append(s('circle', { class: 'ring', cx: r.cx, cy: r.cy, r: 9 }));
    // ring labels: to the right of the ring, nudged apart when rings share a column
    const lab = rings.filter((r) => r.label).sort((p, q) => p.cy - q.cy);
    const placed = [];
    for (const r of lab) {
      const left = r.cx + 60 > W - m.r;
      let y = r.cy + 4;
      for (const p of placed) if (Math.abs(p.cx - r.cx) < 40 && Math.abs(p.y - y) < 13) y = p.y + 13;
      placed.push({ cx: r.cx, y });
      svg.append(s('text', { class: 'rlab', x: left ? r.cx - 13 : r.cx + 13, y, 'text-anchor': left ? 'end' : 'start' }, r.label));
    }
    return svg;
  }

  // ------------------------------------------------------------------ data loading
  function load() {
    const embedded = () => { doc.documentElement.setAttribute('data-source', 'embedded'); return JSON.parse($('#data-embedded').textContent); };
    if (location.protocol === 'file:') return Promise.resolve(embedded());   // fetch() of a local file is blocked in browsers
    return fetch('data.json', { cache: 'no-cache' })
      .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then((j) => { doc.documentElement.setAttribute('data-source', 'data.json'); return j; })
      .catch(() => embedded());
  }

  function bindStatic() {
    for (const el of $$('[data-t]')) {
      const v = get(el.dataset.t);
      el.textContent = typeof v === 'string' ? v : v && v.v != null ? vtext(v) : '';
    }
  }

  // ------------------------------------------------------------------ header readout
  function renderReadout() {
    const P = D.machine.parts;
    const mb = P.xchg.mailbox, sz = P.xchg.sizes;
    const rng = { v: [mb.v[0], mb.v[mb.v.length - 1]], range: true, u: mb.u };
    const items = [
      ['gpu', 'GPU memory', vnode(P.gpu.measured), ['per card · spec ', vtext(P.gpu.spec)]],
      ['ram', 'RAM read', vnode(P.ram.measured), ['by the CPU · spec ', vtext(P.ram.spec), ' (C)']],
      ['pcie', 'PCIe copy', vnode(P.pcie.measured), ['per link · spec ', vtext(P.pcie.spec), ' (C)']],
      ['pcie', 'GPU ↔ GPU', vnode(rng), ['no P2P · via host · ', fv(pick(sz, 0)), '-', fv(pick(sz, sz.v.length - 1)), NB, sz.u]],
      [null, 'NVMe read', vnode(P.nvme.measured), ['sequential']],
    ];
    $('#readout').replaceChildren(...items.map(([c, k, v, sub]) => h('div', { class: 'ro' },
      h('div', { class: 'ro-k' }, h('i', { style: c ? `--c:${RES[c]}` : '--c:var(--ink-2)' }), k),
      h('div', { class: 'ro-v' }, v), h('div', { class: 'ro-s' }, sub))));
  }

  // ------------------------------------------------------------------ 1. machine
  const V1 = {};
  function buildMachine(view) {
    const P = D.machine.parts;
    const pick1 = (part) => () => { S.part = part; renderMachine(); writeHash(); };
    const btn = (cls, part, label, kids) => h('button', { class: cls, type: 'button', 'data-part': part, 'aria-pressed': 'false', 'aria-label': label, onclick: pick1(part) }, kids);
    const nk = (t, c) => h('div', { class: 'node-k' }, h('i', { style: c ? `--c:${RES[c]}` : null }), t);
    const nv = (o, spec) => h('div', { class: 'node-v' }, vnode(o), spec ? h('span', { class: 'spec' }, spec) : null);
    const wire = (c) => h('span', { class: 'link-wire', style: `--c:${RES[c]}`, 'aria-hidden': 'true' });
    const mb = P.xchg.mailbox, sz = P.xchg.sizes;
    const host = h('div', { class: 'host' }, h('div', { class: 'host-l' }, 'Host'),
      btn('node', 'cpu', 'CPU details', [nk('CPU'), h('div', { class: 'node-t' }, P.cpu.label), h('div', { class: 'node-d' }, P.cpu.spec)]),
      btn('node', 'ram', 'RAM details', [nk('RAM', 'ram'), h('div', { class: 'node-t' }, P.ram.label), nv(P.ram.measured, ['read by the CPU · spec ', vtext(P.ram.spec), ' (C)'])]));
    const gpu = (i) => btn('node gpu' + i, 'gpu', `GPU ${i} details`, [nk('GPU ' + i, 'gpu'), h('div', { class: 'node-t' }, P.gpu.label.split(',')[0]),
      nv(P.gpu.measured, ['memory, measured roof · spec ', vtext(P.gpu.spec)])]);
    const link = (i) => btn('link l' + i, 'pcie', `PCIe link ${i} details`, [wire('pcie'), h('span', { class: 'link-body' },
      h('div', { class: 'link-t' }, P.pcie.label.split(',')[0]), h('div', { class: 'link-v' }, vnode(P.pcie.measured)),
      h('div', { class: 'link-d' }, 'copy engine, host to GPU ' + i))]);
    const xg = btn('link xg', 'xchg', 'GPU-to-GPU exchange details', [wire('pcie'), h('span', { class: 'link-body' },
      h('div', { class: 'link-t' }, 'GPU ⇄ GPU through host memory'),
      h('div', { class: 'link-v' }, vnode({ v: [mb.v[0], mb.v[mb.v.length - 1]], range: true, u: mb.u })),
      h('div', { class: 'link-d' }, ['mailbox exchange, ', fv(pick(sz, 0)), '-', fv(pick(sz, sz.v.length - 1)), NB, sz.u]))]);
    const nop = h('div', { class: 'nop', 'aria-hidden': 'true' }, h('span', null, h('b', null, '✕'), ' no P2P'));
    const nvme = btn('node nvme', 'nvme', 'Storage details', [nk('Storage'), h('div', { class: 'node-t' }, P.nvme.label), nv(P.nvme.measured, 'sequential read')]);
    const diagram = h('div', { class: 'mach', role: 'group', 'aria-label': 'Machine diagram: select a part' }, host, link(0), gpu(0), xg, nop, link(1), gpu(1), nvme);
    V1.detail = h('div', { class: 'detail', 'aria-live': 'polite' });
    view.replaceChildren(diagram, V1.detail);
    V1.diagram = diagram;
  }
  function railFill(max, val, c) {
    return h('div', { class: 'rail', style: `--c:${c}` }, h('div', { class: 'fill', style: `width:${Math.min(100, (val / max) * 100)}%` }));
  }
  function railRange(max, lo, hi, c) {
    return h('div', { class: 'rail', style: `--c:${c}` },
      h('div', { class: 'fill', style: `width:${(lo / max) * 100}%` }),
      h('div', { class: 'rng', style: `left:${(lo / max) * 100}%;width:${((hi - lo) / max) * 100}%;opacity:.55` }));
  }
  function renderMachine() {
    const P = D.machine.parts;
    if (!P[S.part]) S.part = DEF.part;
    for (const b of $$('[data-part]', V1.diagram)) b.setAttribute('aria-pressed', String(b.dataset.part === S.part));
    const p = P[S.part];
    const col = { gpu: RES.gpu, ram: RES.ram, pcie: RES.pcie, xchg: RES.pcie, nvme: 'var(--ink-2)', cpu: 'var(--ink-2)' }[S.part];
    const left = h('div', null, h('h3', { class: 'dt-h' }, p.name), h('p', { class: 'dt-s' }, p.label));
    if (p.spec && p.measured && typeof p.spec === 'object') {
      const max = p.spec.v;
      left.append(h('div', { class: 'gauge' },
        h('div', { class: 'gauge-top' },
          h('div', { class: 'gauge-m' }, vnode(p.measured), h('div', { class: 'gauge-n', style: 'margin:2px 0 0' }, 'measured: ' + p.measured.note)),
          h('div', { class: 'gauge-s' }, h('em', null, 'spec sheet'), vnode(p.spec))),
        railFill(max, p.measured.v, col),
        h('div', { class: 'gauge-n' }, 'Track: spec-sheet figure. ' + (p.spec.note || ''), p.spec.c ? cmark() : null)));
      if (p.readings) {
        left.append(h('div', { class: 'readings' }, p.readings.map((r) => {
          const x = r.x, isR = Array.isArray(x.v);
          return h('div', { class: 'rd' }, h('div', { class: 'rd-l' }, r.label),
            isR ? railRange(max, Math.min(...x.v), Math.max(...x.v), col) : railFill(max, x.v, col),
            h('div', { class: 'rd-v' }, vnode(x)));
        })));
      }
    } else if (S.part === 'xchg') {
      const mx = Math.max(...p.nccl.v);
      left.append(h('div', { class: 'legend' },
        h('span', null, h('span', { class: 'sw', style: `--c:${RES.pcie}` }), 'engine mailbox, inside a CUDA graph'),
        h('span', null, h('span', { class: 'sw', style: '--c:var(--faint)' }), 'NCCL all-reduce, P2P disabled')));
      left.append(h('div', { class: 'xbars' }, p.sizes.v.map((sz, i) => h('div', { class: 'xb' },
        h('div', { class: 'xb-k' }, num(sz), h('small', null, NB + p.sizes.u)),
        h('div', { class: 'bar', style: `--c:${RES.pcie}` }, h('div', { class: 'fill', style: `width:${(p.mailbox.v[i] / mx) * 78}%` }),
          h('b', { style: `left:${(p.mailbox.v[i] / mx) * 78}%` }, fv(p.mailbox, i), h('small', null, p.mailbox.u))),
        h('div', { class: 'bar', style: '--c:var(--faint)' }, h('div', { class: 'fill', style: `width:${(p.nccl.v[i] / mx) * 78}%` }),
          h('b', { style: `left:${(p.nccl.v[i] / mx) * 78}%` }, fv(p.nccl, i), h('small', null, p.nccl.u)))))));
      left.append(h('p', { class: 'gauge-n' }, p.mailbox.note + '. ' + p.nccl.note + '.'));
    } else if (p.measured) {
      left.append(h('div', { class: 'gauge' }, h('div', { class: 'gauge-m' }, vnode(p.measured)), h('div', { class: 'gauge-n' }, p.measured.note + '; no spec-sheet figure in the report.')));
    } else {
      left.append(h('p', { class: 'gauge-n' }, 'No ceiling of the CPU itself is measured in the report (Table 1: —). The RAM ceiling bounds the CPU expert compute.'));
    }
    const right = h('div', null, h('div', { class: 'hd' }, 'Engine parts on it'),
      h('ul', { class: 'roles' }, p.roles.v.map((r) => h('li', null, r))));
    if (p.extra || p.capacity) {
      const kv = h('dl', { class: 'kv' });
      for (const e of p.extra || []) kv.append(h('dt', null, e.label), h('dd', null, vnode(e.x)));
      if (p.capacity) kv.append(h('dt', null, p.capacity.note), h('dd', null, vnode(p.capacity)));
      right.append(kv);
    }
    right.append(h('p', { class: 'src', style: 'margin-top:12px' }, [p.source, p.roles.source].join(' · ')));
    V1.detail.replaceChildren(left, right);
  }

  // ------------------------------------------------------------------ 2. token path
  const V2 = {};
  const pathModel = () => D.path.models.find((m) => m.id === S.m2) || D.path.models[0];
  const gpuWord = (g) => `${g} ${plural(+g, 'GPU', 'GPUs')}`;
  function buildPath(view) {
    V2.model = chips('Model', D.path.models.map((m) => ({ value: m.id, label: m.label })), S.m2, (v) => { S.m2 = v; renderPath(); writeHash(); });
    V2.gpus = chips('GPUs', [{ value: '1', label: '1 GPU' }, { value: '2', label: '2 GPUs · TP 2' }], S.g2, (v) => { S.g2 = v; renderPath(); writeHash(); });
    V2.out = h('div');
    view.replaceChildren(h('div', { class: 'ctls' }, V2.model.el, V2.gpus.el), V2.out);
  }
  function renderPath() {
    const m = pathModel();
    S.m2 = m.id;
    if (!m.configs[S.g2]) S.g2 = Object.keys(m.configs).sort().reverse()[0];
    V2.model.set(S.m2); V2.gpus.set(S.g2);
    for (const g of ['1', '2']) V2.gpus.disable(g, !m.configs[g], 'not measured on this model');
    const c = m.configs[S.g2];
    const out = [];
    out.push(h('div', { class: 'p-head' }, h('div', { class: 'p-name' }, m.full), h('div', null,
      m.unreleased ? h('span', { class: 'tag unrel' }, 'unreleased port') : null, ' ', h('span', { class: 'tag' }, gpuWord(S.g2)))));
    const bytes = [];
    if (c.bytes) bytes.push(`Reads ${vtext(c.bytes)} per token`, S.g2 === '2' ? [', ', vtext(c.per_card), ' per card'] : null, c.per_card.c ? cmark() : null, '. ');
    else bytes.push('GPU bytes ', vtext(c.per_card), S.g2 === '2' ? ' per card' : ' on the card', c.per_card.c ? cmark() : null, '. ');
    if (c.bytes_note) bytes.push(c.bytes_note[0].toUpperCase() + c.bytes_note.slice(1) + '.');
    out.push(h('p', { class: 'p-bytes' }, bytes));
    const fl = (key, col, floor, share, what) => h('div', { class: 'fl' },
      h('div', { class: 'fl-k' }, h('i', { style: `--c:${col}` }), what),
      h('div', { class: 'fl-r' },
        h('div', { class: 'rail', style: `--c:${col}`, role: 'img', 'aria-label': `${what}: ${vtext(floor)} of a ${vtext(c.step)} step, ${vtext(share)}` },
          h('div', { class: 'fill', style: `width:${(floor.v / c.step.v) * 100}%` })),
        h('div', { class: 'fl-labs' }, h('span', null, 'floor ', h('b', null, vtext(floor)), floor.c ? cmark() : null),
          h('span', null, 'whole step ', h('b', null, vtext(c.step)), c.step.c ? cmark() : null))),
      h('div', { class: 'fl-p' }, fv(share), h('small', null, '%')));
    out.push(h('div', { class: 'floors' },
      h('div', { class: 'hd' }, 'Step against its floor', h('small', null, 'no drafter, one request · Table 5, Figure 1a')),
      fl('g', RES.gpu, c.floor_gpu, c.share_gpu, 'GPU floor, at ' + vtext(D.machine.parts.gpu.measured)),
      c.floor_ram ? fl('r', RES.ram, c.floor_ram, c.share_ram, 'RAM floor of the missed experts, at ' + vtext(D.machine.parts.ram.measured)) : null));
    out.push(h('div', { class: 'hd' }, 'The four engine parts', h('small', null, 'in the order of the report, §2-§5')));
    out.push(h('div', { class: 'stations' }, D.path.stages.map((stg, i) => {
      const x = c.stages[stg.id];
      const na = x.state === 'na';
      return h('div', { class: 'st' + (na ? ' na' : '') },
        h('div', { class: 'st-n' }, `${stg.sec} · part ${i + 1} of ${D.path.stages.length}`),
        h('div', { class: 'st-t' }, stg.name),
        h('div', { class: 'st-r' }, stg.res.map((r) => h('span', { class: 'sw', style: `--c:${RES[r]}`, title: RESNAME[r] })), stg.resource),
        h('div', { class: 'st-c' }, 'ceiling ', vnode(stg.ceiling)),
        x.meter ? h('div', { class: 'meter' },
          h('div', { class: 'meter-v' }, h('b', null, fv(x.meter), h('small', null, '%')), h('span', null, x.meter_label)),
          railFill(100, x.meter.v, RES[x.meter_res])) : null,
        h('ul', null, x.lines.map((l) => h('li', null, l))),
        h('div', { class: 'src' }, x.source));
    })));
    // overview: every model (Figure 1a) and every part while it runs (Figure 1b)
    const rows = [];
    for (const mm of D.path.models) for (const g of Object.keys(mm.configs).sort().reverse()) rows.push([mm, g, mm.configs[g]]);
    const mx = 100;
    const am = h('div', null, h('div', { class: 'hd' }, 'Every model', h('small', null, 'floor ÷ step · Figure 1a')),
      h('div', { class: 'legend' }, h('span', null, h('span', { class: 'sw', style: `--c:${RES.gpu}` }), 'GPU floor'), h('span', null, h('span', { class: 'sw', style: `--c:${RES.ram}` }), 'RAM floor (Flash-Next)')),
      h('div', { class: 'am-list' }, rows.map(([mm, g, cc]) => h('button', {
        class: 'am-row', type: 'button', 'data-fkey': `am-${mm.id}-${g}`, 'aria-current': String(mm.id === S.m2 && g === S.g2),
        onclick: () => { S.m2 = mm.id; S.g2 = g; keepFocus(renderPath); writeHash(); },
      }, h('span', { class: 'am-l' }, `${mm.label} · ${gpuWord(g)}`),
      h('span', { class: 'am-b' },
        h('span', { class: 'bar', style: `--c:${RES.gpu}` }, h('span', { class: 'fill', style: `width:${(cc.share_gpu.v / mx) * 82}%` }), h('b', { style: `left:${(cc.share_gpu.v / mx) * 82}%` }, fv(cc.share_gpu), '%')),
        cc.share_ram ? h('span', { class: 'bar', style: `--c:${RES.ram}` }, h('span', { class: 'fill', style: `width:${(cc.share_ram.v / mx) * 82}%` }), h('b', { style: `left:${(cc.share_ram.v / mx) * 82}%` }, fv(cc.share_ram), '%')) : null)))));
    const f1b = [];
    for (const mm of D.path.models) for (const g of Object.keys(mm.configs)) for (const stg of D.path.stages) {
      const x = mm.configs[g].stages[stg.id];
      if (x.meter) f1b.push({ mm, g, x, on: mm.id === S.m2 && g === S.g2 });
    }
    const fb = h('div', { class: 'fig1b' }, h('div', { class: 'hd' }, 'Each part while it runs', h('small', null, 'rate ÷ ceiling · Figure 1b')),
      f1b.map(({ mm, g, x, on }) => h('div', { class: 'f1b-row' + (on ? ' on' : '') },
        h('span', { class: 'am-l' }, `${x.meter_short[0].toUpperCase()}${x.meter_short.slice(1)} · ${mm.label}`),
        h('span', { class: 'bar', style: `--c:${RES[x.meter_res]}` }, h('span', { class: 'fill', style: `width:${x.meter.v * 0.85}%` }), h('b', { style: `left:${x.meter.v * 0.85}%` }, fv(x.meter), '%')))),
      h('p', { class: 'gauge-n' }, 'Highlighted: the parts measured on the selected model. Ceilings: GPU memory ', vtext(D.machine.parts.gpu.measured), ' per card, RAM ', vtext(D.machine.parts.ram.measured), ', PCIe copy engine ', vtext(D.machine.parts.pcie.measured), ' per link.'));
    out.push(h('div', { class: 'allmodels' }, am, fb));
    V2.out.replaceChildren(...out);
  }

  // ------------------------------------------------------------------ 3. speculative decoding
  const V3 = {};
  function buildSpec(view) {
    V3.model = chips('Model', D.spec.models.map((m) => ({ value: m.id, label: m.label })), S.m3, (v) => { S.m3 = v; renderSpec(); writeHash(); });
    V3.out = h('div');
    view.replaceChildren(h('div', { class: 'ctls' }, V3.model.el), V3.out);
  }
  function renderSpec() {
    const m = D.spec.models.find((x) => x.id === S.m3) || D.spec.models[0];
    S.m3 = m.id; V3.model.set(S.m3);
    let max = 0;
    for (const mm of D.spec.models) for (const g of mm.groups) for (const r of g.rows) max = Math.max(max, r.tps.v);
    const SC = max / 0.8;
    const ticks = yTicks(max, 5).filter((t) => t <= SC);
    const out = [];
    out.push(h('div', { class: 'p-head' }, h('div', { class: 'p-name' }, m.label), m.unreleased ? h('span', { class: 'tag unrel' }, 'unreleased port') : null));
    const colsSeen = [...new Set(m.groups.flatMap((g) => g.rows.map((r) => r.cols)))].sort((x, y) => x - y);
    out.push(h('div', { class: 'legend' },
      colsSeen.map((c) => h('span', null, h('span', { class: 'sw', style: `--c:${wcol(c)}` }),
        c === 0 ? 'columns vary by round' : `${c} ${plural(c, 'column', 'columns')} per round`, c === 1 ? ' (no drafter)' : '')),
      h('span', null, h('span', { class: 'tag' }, 'default'), 'launcher default')));
    for (const g of m.groups) {
      out.push(h('div', { class: 'sp-group' }, h('div', { class: 'gpus-h' }, gpuWord(g.gpus)),
        h('div', { class: 'sp-grid', role: 'table', 'aria-label': `${m.label}, ${gpuWord(g.gpus)}: decode of one request` },
          h('div', { class: 'sp-row sp-head', role: 'row' },
            h('div', { class: 'sp-h', role: 'columnheader' }, 'drafter'),
            h('div', { class: 'sp-h sp-bar', role: 'columnheader' }, h('span', { class: 'sp-u' }, 'decode, tok/s'), h('div', { class: 'sp-axis' }, ticks.map((t) => h('span', { class: 'tick', style: `left:${(t / SC) * 100}%` }, String(t))))),
            ['vs none', 'accept.', 'tok / round', 'ms / round'].map((t) => h('div', { class: 'sp-h sp-num', role: 'columnheader' }, t))),
          g.rows.map((r) => h('div', { class: 'sp-row', role: 'row' },
            h('div', { class: 'sp-n', role: 'cell' }, r.name, r.default ? h('span', { class: 'tag' }, 'default') : null, r.note ? h('span', { class: 'win-acc', style: 'flex-basis:100%' }, r.note) : null),
            h('div', { class: 'sp-bar', role: 'cell' }, h('div', { class: 'bar', style: `--c:${wcol(r.cols)}` },
              h('div', { class: 'fill', style: `width:${(r.tps.v / SC) * 100}%` }),
              h('b', { style: `left:${(r.tps.v / SC) * 100}%` }, fv(r.tps), h('small', null, r.tps.u)))),
            h('div', { class: 'sp-num', role: 'cell', 'data-l': 'vs none' }, r.gain ? ['+', fv(r.gain), '%'] : h('span', { class: 'dim' }, '—')),
            h('div', { class: 'sp-num' + (r.acc ? '' : ' dim'), role: 'cell', 'data-l': 'acceptance' }, r.acc ? fv(r.acc) : '—'),
            h('div', { class: 'sp-num' + (r.tpr ? '' : ' dim'), role: 'cell', 'data-l': 'tok / round' }, r.tpr ? [fv(r.tpr), r.tpr.c ? cmark() : null] : '—'),
            h('div', { class: 'sp-num' + (r.msr ? '' : ' dim'), role: 'cell', 'data-l': 'ms / round' }, r.msr ? [fv(r.msr), r.msr.c ? cmark() : null] : '—'))))));
    }
    out.push(h('ul', { class: 'notes' }, m.notes.v.map((n) => h('li', null, n))));
    out.push(h('p', { class: 'src', style: 'margin-top:10px' }, [...new Set([D.spec.source, ...m.notes.source.split(', ')])].join(', ')));
    V3.out.replaceChildren(...out);
  }

  // ------------------------------------------------------------------ 4. concurrency
  const V4 = {};
  const concModel = () => D.conc.models.find((m) => m.id === S.m4) || D.conc.models[0];
  function buildConc(view) {
    const NS4 = D.conc.clients.v;
    V4.model = chips('Model, 2 GPUs', D.conc.models.map((m) => ({ value: m.id, label: m.label })), S.m4, (v) => { S.m4 = v; renderConc(); writeHash(); });
    V4.set = chips('Prompts', [{ value: 'prose', label: 'prose' }, { value: 'code', label: 'code rewriting' }], S.p4, (v) => { S.p4 = v; renderConc(); writeHash(); });
    V4.n = slider('Concurrent clients', NS4.length, S.n4, (i) => [String(NS4[i]), h('small', null, plural(NS4[i], 'client', 'clients'))],
      NS4.map(String), (i) => { S.n4 = i; renderConc(); writeHash(); });
    V4.x = chips('Chart against', [{ value: 'clients', label: 'clients' }, { value: 'columns', label: 'columns per round' }], S.x4, (v) => { S.x4 = v; renderConc(); writeHash(); });
    V4.out = h('div');
    view.replaceChildren(h('div', { class: 'ctls' }, V4.model.el, V4.set.el, V4.n.el, V4.x.el), V4.out);
  }
  function concState() {
    const m = concModel();
    S.m4 = m.id;
    if (!m.sets[S.p4]) S.p4 = 'prose';
    S.n4 = Math.max(0, Math.min(D.conc.clients.v.length - 1, S.n4));
    if (S.x4 !== 'columns') S.x4 = 'clients';
    return m;
  }
  const colsAt = (w, i) => (Array.isArray(w.cols) ? w.cols[i] : w.cols);
  function renderConc() {
    const m = concState();
    V4.model.set(S.m4); V4.set.set(S.p4); V4.n.set(S.n4); V4.x.set(S.x4);
    V4.set.disable('code', !m.sets.code, 'code prompts measured on the 27B NVFP4 only');
    const set = m.sets[S.p4], W = set.windows, NS4 = D.conc.clients.v, i = S.n4, N = NS4[i];
    const tol = (m.family === 'flashnext' ? D.conc.tie.flashnext : D.conc.tie.dense);
    const fast = m.fastpath ? m.fastpath.v : null;
    const bestAt = (j) => W.reduce((b, w) => (w.thr.v[j] > b.thr.v[j] ? w : b), W[0]);
    const best = bestAt(i);
    const isTie = (w, j) => w !== bestAt(j) && ((bestAt(j).thr.v[j] - w.thr.v[j]) / bestAt(j).thr.v[j]) * 100 <= tol.v;
    let mx = 0;
    for (const w of W) mx = Math.max(mx, ...w.thr.v);
    const SC = mx / (innerWidth < 700 ? 0.66 : 0.74);
    const setLabel = S.p4 === 'code' ? 'code rewriting' : 'prose';

    const left = h('div', null,
      h('div', { class: 'cc-n' }, `${N} ${plural(N, 'client', 'clients')} · ${m.label} · ${setLabel} `, m.unreleased ? h('span', { class: 'tag unrel' }, 'unreleased port') : null),
      h('p', { class: 'cc-sub' }, 'Throughput of each draft window; bar scale fixed across client counts. Squares: the columns one round verifies, one group per request',
        fast ? ['; hollow squares: more than ', String(fast), ' columns per round, past the small-batch kernel of the 3-bit linear layers (read in the source, not profiled)'] : '', '.'),
      W.map((w) => {
        const cols = colsAt(w, i), K = cols - 1, tot = N * cols, top = w === best, tie = isTie(w, i);
        const reqs = [];
        let k = 0;
        for (let r = 0; r < N; r++) {
          const sq = [];
          for (let c = 0; c < cols; c++, k++) sq.push(h('i', { class: fast && k >= fast ? 'over' : null }));
          reqs.push(h('span', { class: 'req' }, sq));
        }
        return h('div', { class: 'win' },
          h('div', { class: 'win-h' }, glyph(Math.max(...[].concat(w.cols)), wcol(Math.max(...[].concat(w.cols)))), h('span', { class: 'win-name' }, w.name),
            w.tag ? h('span', { class: 'tag' }, w.tag) : w.default ? h('span', { class: 'tag' }, 'default') : null,
            top ? h('span', { class: 'tag hi' }, 'highest') : null,
            tie ? h('span', { class: 'tag tie' }, '≈ tie, within ' + vtext(tol)) : null,
            w.acc ? h('span', { class: 'win-acc' }, Array.isArray(w.acc.v) ? `acceptance at ${N} ${plural(N, 'client', 'clients')} ${fv(w.acc, i)}` : 'acceptance over N ' + fv(w.acc)) : null,
            w.drafts ? h('span', { class: 'win-acc' }, `· ${fv(w.drafts, i)} ${w.drafts.u}`) : null),
          h('div', { class: 'win-b' },
            h('div', { class: 'bar', style: `--c:${wcol(cols)}`, role: 'img', 'aria-label': `${w.name}: ${vtext(w.thr, i)}` },
              h('div', { class: 'fill' + (top ? ' best' : ''), style: `width:${(w.thr.v[i] / SC) * 100}%` }),
              h('b', { style: `left:${(w.thr.v[i] / SC) * 100}%` }, fv(w.thr, i), h('small', null, 'tok/s'))),
            h('span', { class: 'win-rate' }, 'per request ' + fv(w.rate, i))),
          h('div', { class: 'cols', style: `--c:${wcol(cols)}` }, reqs,
            h('span', { class: 'cols-t' }, `${N} × (${K} + 1) = `, h('b', null, String(tot)), ` ${plural(tot, 'column', 'columns')}`,
              fast && tot > fast ? ` · more than ${fast}` : '')));
      }));

    // chart
    const byCols = S.x4 === 'columns';
    const xs = (w, j) => (byCols ? NS4[j] * colsAt(w, j) : NS4[j]);
    let xmax = 0;
    for (const w of W) for (let j = 0; j < NS4.length; j++) xmax = Math.max(xmax, xs(w, j));
    const xt = [];
    for (let v = 1; v <= Math.max(xmax, 8); v *= 2) xt.push({ v, l: String(v) });
    const yt = yTicks(mx, 4);
    const series = W.map((w) => ({
      color: wcol(Math.max(...[].concat(w.cols))), marker: wstyle(Math.max(...[].concat(w.cols))).marker, dash: wstyle(Math.max(...[].concat(w.cols))).dash, emph: true,
      pts: NS4.map((n, j) => ({ x: xs(w, j), y: w.thr.v[j], big: j === i,
        tip: `${fv(w.thr, j)} tok/s\n${w.name} · ${n} ${plural(n, 'client', 'clients')}\n${n * colsAt(w, j)} columns per round · per request ${fv(w.rate, j)}` })),
    }));
    const rings = NS4.map((n, j) => { const b = bestAt(j); return { x: xs(b, j), y: b.thr.v[j], label: byCols ? `N=${n}` : null }; });
    const cw = Math.max(280, (V4.out.clientWidth || 900) * (innerWidth > 860 ? 0.44 : 1) - 4);
    const chart = lineChart({
      w: cw, h: 270, xlog: true, xdom: [0.8, Math.max(xmax, 8) * 1.25], xticks: xt, yticks: yt, series, rings,
      cursorX: byCols ? null : N, shadeFrom: byCols && fast ? fast * 1.41 : null, shadeLabel: byCols && fast ? `more than ${fast} columns` : null,
      xlabel: byCols ? 'columns per round, clients × (drafts + 1)' : 'concurrent clients',
      aria: `Throughput against ${byCols ? 'columns per round' : 'clients'} for every draft window of ${m.label}, ${setLabel}`,
    });
    const right = h('div', null,
      h('div', { class: 'chart-cap' }, byCols ? 'throughput, tok/s · ring: highest at the client count N written beside it' : 'throughput, tok/s · ring: highest at each N'),
      h('div', { class: 'legend' }, W.map((w) => h('span', null, glyph(Math.max(...[].concat(w.cols)), wcol(Math.max(...[].concat(w.cols)))), w.name))),
      h('div', { class: 'chart' }, chart),
      byCols ? h('p', { class: 'gauge-n' }, 'Each point sits at its own column count, so the windows spread along the axis; the selected client count is drawn larger.') : null);

    const rule = h('div', { class: 'rule' },
      h('div', { class: 'formula' }, 'columns per round = clients × (drafts + 1)'),
      h('div', null, D.conc.function),
      h('div', null, h('b', null, 'Tie rule. '), D.conc.tie.text, ' This model: ', vtext(tol), '.'));
    const notes = h('ul', { class: 'notes' }, set.notes.v.map((n) => h('li', null, n)), h('li', null, D.conc.general.v));
    V4.out.replaceChildren(h('div', { class: 'cc' }, left, right), rule, notes, h('p', { class: 'src', style: 'margin-top:10px' }, [D.conc.source, set.notes.source, D.conc.tie.source].join(' · ')));
  }

  // ------------------------------------------------------------------ 5. long prompts
  const V5 = {};
  const ctxLabel = (m) => `${m.label} · ${gpuWord(m.gpus)}`;
  const CTXH = ['model · GPUs · drafter', 'prompt tokens', 'prefill tok/s', 'prompt time, s', 'decode tok/s', 'decode sd over the sweep'];
  function buildCtx(view) {
    const st = D.ctx.stops.v;
    V5.L = slider('Prompt length', st.length, S.L5, (i) => ['≈ ', st[i], h('small', null, 'tokens')], st, (i) => { S.L5 = i; renderCtx(); writeHash(); });
    V5.model = chips('Follow', D.ctx.models.map((m) => ({ value: m.id, label: ctxLabel(m) })), S.m5, (v) => { S.m5 = v; renderCtx(); writeHash(); });
    V5.out = h('div');
    view.replaceChildren(h('div', { class: 'ctls' }, V5.L.el, V5.model.el), V5.out);
  }
  function renderCtx() {
    const M = D.ctx.models, st = D.ctx.stops.v;
    const m = M.find((x) => x.id === S.m5) || M[0];
    S.m5 = m.id; S.L5 = Math.max(0, Math.min(st.length - 1, S.L5));
    V5.L.set(S.L5); V5.model.set(S.m5);
    const i = S.L5, has = i < m.tokens.v.length;
    const atMax = has && m.maxctx && i === m.tokens.v.length - 1;
    const tiles = has ? [
      ['prompt tokens', fv(m.tokens, i), '', atMax ? `longest prompt within the launcher's maximum context (${vtext(m.maxctx)})` : `stop ≈ ${st[i]}`],
      ['prefill', fv(m.prefill, i), 'tok/s', i === 0 && m.prefill_sd0 ? `sd ${fv(m.prefill_sd0)} between rounds` : 'prompt tokens ÷ prompt time'],
      ['time to first token', fv(m.time, i), 's', "the server's prompt time"],
      ['decode', fv(m.decode, i), 'tok/s', `${m.drafter} · sd ${fv(m.sd)} tok/s over the sweep`],
    ] : [
      ['prompt tokens', '—', '', `beyond this launcher's maximum context: ${vtext(m.maxctx)}`], ['prefill', '—', '', ''], ['time to first token', '—', '', ''], ['decode', '—', '', ''],
    ];
    const head = h('div', { class: 'p-head' }, h('div', { class: 'p-name' }, ctxLabel(m) + (m.mark || '')),
      h('div', null, m.unreleased ? h('span', { class: 'tag unrel' }, 'unreleased port') : null, ' ', h('span', { class: 'tag' }, m.drafter)));
    const top = h('div', { class: 'cx-top', 'aria-live': 'polite' }, tiles.map(([k, v, u, sub]) => h('div', { class: 'tile' },
      h('div', { class: 'tile-k' }, k), h('div', { class: 'tile-v' }, v, u ? h('small', null, u) : null), h('div', { class: 'tile-s' }, sub))));
    const cw = Math.max(280, ((V5.out.clientWidth || 900) - (innerWidth > 860 ? 26 : 0)) / (innerWidth > 860 ? 2 : 1));
    const mk = (key, max) => {
      const series = M.map((mm) => ({
        color: mm.id === m.id ? 'var(--ink)' : 'var(--faint)', width: mm.id === m.id ? 2.25 : 1.5, emph: mm.id === m.id, r: mm.id === m.id ? 4 : 2.5,
        pts: mm.tokens.v.map((t, j) => ({ x: t, y: mm[key].v[j], big: j === i && mm.id === m.id,
          tip: `${fv(mm[key], j)} ${mm[key].u}\n${ctxLabel(mm)} · ${mm.drafter}\n${fv(mm.tokens, j)} prompt tokens` })),
      }));
      return lineChart({
        w: cw, h: 230, xlog: true, xdom: [380, 175000], yticks: yTicks(max, 4),
        xticks: [{ v: 1000, l: '1k' }, { v: 10000, l: '10k' }, { v: 100000, l: '100k' }], series,
        cursorX: has ? m.tokens.v[i] : null, xlabel: 'prompt tokens (log scale)',
        aria: `${key === 'prefill' ? 'Prefill rate' : 'Decode rate'} against prompt length for every swept model; ${ctxLabel(m)} highlighted`,
      });
    };
    let pmax = 0, dmax = 0;
    for (const mm of M) { pmax = Math.max(pmax, ...mm.prefill.v); dmax = Math.max(dmax, ...mm.decode.v); }
    const charts = h('div', { class: 'cx-charts' },
      h('div', null, h('div', { class: 'chart-cap' }, 'prefill, tok/s'), h('div', { class: 'chart' }, mk('prefill', pmax))),
      h('div', null, h('div', { class: 'chart-cap' }, 'decode of 256 tokens after the prompt, tok/s'), h('div', { class: 'chart' }, mk('decode', dmax))));
    const leg = h('div', { class: 'legend' }, h('span', null, h('span', { class: 'ln', style: '--c:var(--ink)' }), ctxLabel(m)), h('span', null, h('span', { class: 'ln', style: '--c:var(--faint)' }), 'the other swept models'), h('span', null, 'vertical line: the selected prompt length'));
    const tbl = h('table', { class: 'tbl stack' },
      h('caption', { class: 'chart-cap', style: 'text-align:left;caption-side:top' }, `All swept models at the ≈ ${st[i]} stop · Table 10`),
      h('thead', null, h('tr', null, CTXH.map((t) => h('th', { scope: 'col' }, t)))),
      h('tbody', null, M.map((mm) => {
        const ok = i < mm.tokens.v.length;
        return h('tr', { class: mm.id === m.id ? 'on' : null },
          h('td', null, `${ctxLabel(mm)}${mm.mark || ''} · ${mm.drafter}`, mm.unreleased ? [' ', h('span', { class: 'tag unrel' }, 'unreleased port')] : null),
          ok ? [h('td', { 'data-l': CTXH[1] }, fv(mm.tokens, i), mm.maxctx && i === mm.tokens.v.length - 1 ? h('span', { class: 'dim' }, 'launcher max') : null), h('td', { 'data-l': CTXH[2] }, fv(mm.prefill, i), i === 0 && mm.prefill_sd0 ? h('span', { class: 'dim' }, 'sd ' + fv(mm.prefill_sd0)) : null),
            h('td', { 'data-l': CTXH[3] }, fv(mm.time, i)), h('td', { 'data-l': CTXH[4] }, fv(mm.decode, i)), h('td', { 'data-l': CTXH[5] }, fv(mm.sd))]
            : h('td', { class: 'miss', colspan: 5 }, `beyond the launcher's maximum context (${vtext(mm.maxctx)})`));
      })));
    const notes = h('ul', { class: 'notes' }, D.ctx.notes.v.map((n) => h('li', null, n)));
    // every value of the sweep, without hovering (the grey lines of the charts)
    const full = h('details', { class: 'full' }, h('summary', null, 'All values of the sweep, every model (Table 10)'),
      h('div', { class: 'tbl-wrap', tabindex: '0', role: 'region', 'aria-label': 'Table: the whole context sweep' },
        h('table', { class: 'tbl full-t' },
          h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'model · quantity'), st.map((t) => h('th', { scope: 'col' }, '≈ ' + t)))),
          h('tbody', null, M.map((mm) => [
            h('tr', { class: 'grp' + (mm.id === m.id ? ' on' : '') }, h('td', { colspan: st.length + 1 }, `${ctxLabel(mm)}${mm.mark || ''} · ${mm.drafter}`)),
            [['prompt tokens', 'tokens'], ['prefill, tok/s', 'prefill'], ['prompt time, s', 'time'], ['decode, tok/s', 'decode']].map(([lab, key]) =>
              h('tr', null, h('td', null, lab), st.map((_, j) => h('td', null, j < mm[key].v.length ? fv(mm[key], j) : '—')))),
          ])))));
    V5.out.replaceChildren(head, top, leg, charts, h('div', { class: 'tbl-wrap', tabindex: '0', role: 'region', 'aria-label': 'Table: all swept models at the selected prompt length' }, tbl), notes,
      full, h('p', { class: 'src', style: 'margin-top:10px' }, [D.ctx.source, D.ctx.notes.source].join(' · ')));
  }

  // ------------------------------------------------------------------ 6. reference
  function renderMemory(view) {
    const M = D.memory;
    view.replaceChildren(h('div', { class: 'tbl-wrap', tabindex: '0', role: 'region', 'aria-label': 'Table: memory and start-up' },
      h('table', { class: 'tbl stack' }, h('thead', null, h('tr', null, M.head.map((t) => h('th', { scope: 'col' }, t)))),
        h('tbody', null, M.rows.map((r) => h('tr', null, r.v.map((c, j) => h('td', { 'data-l': M.head[j] }, c))))))),
    h('ul', { class: 'notes' }, M.notes.v.map((n) => h('li', null, n))));
  }
  function renderLlama(view) {
    const L = D.llama;
    const grp = (name, rows) => {
      let mx = 0;
      for (const r of rows) mx = Math.max(mx, r.capy.v, r.llama.v);
      const bar = (o, sd, col) => h('div', { class: 'bar', style: `--c:${col}` }, h('div', { class: 'fill', style: `width:${(o.v / mx) * 62}%` }),
        h('b', { style: `left:${(o.v / mx) * 62}%` }, fv(o), h('small', null, ' ± ' + fv(sd) + ' ' + o.u)));
      return h('div', null, h('div', { class: 'll-cap' }, name), rows.map((r) => h('div', { class: 'll-row' }, h('div', { class: 'am-l' }, r.label),
        bar(r.capy, r.capy_sd, 'var(--ink)'), bar(r.llama, r.llama_sd, 'var(--faint)'))));
    };
    view.replaceChildren(
      h('div', { class: 'legend' }, h('span', null, h('span', { class: 'sw', style: '--c:var(--ink)' }), 'capy-flash, development build of 21 Nov'), h('span', null, h('span', { class: 'sw', style: '--c:var(--faint)' }), 'llama.cpp')),
      grp('decode', L.rows.filter((r) => r.group === 'decode')), grp('prefill', L.rows.filter((r) => r.group === 'prefill')));
  }
  function renderLimits(view) { view.replaceChildren(h('ol', { class: 'lim' }, D.limits.v.map((l) => h('li', null, l)))); }
  function renderMethod(view) { view.replaceChildren(h('dl', { class: 'meth' }, D.method.v.map(([k, v]) => h('div', null, h('dt', null, k), h('dd', null, v))))); }
  function renderDownloads(view) {
    view.replaceChildren(...D.downloads.v.map((d) => h('a', { class: 'dlb', href: d.href, download: d.href.endsWith('.json') ? 'capy-flash-v0.16.6-data.json' : null },
      h('span', { class: 'dlb-i', 'aria-hidden': 'true' }, d.href.endsWith('.pdf') ? 'PDF' : 'JSON'), h('span', { class: 'dlb-t' }, d.label), h('span', { class: 'dlb-d' }, d.desc))));
  }

  // ------------------------------------------------------------------ audit (#audit=1): render every state and list its text, for number checks
  function textOf(el) {
    const parts = [], ticks = [];
    const w = doc.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        const p = n.parentElement;
        if (!p || p.closest('script,style,#audit-out,noscript')) return NodeFilter.FILTER_REJECT;
        if (p.closest('.tick')) { ticks.push(n.nodeValue); return NodeFilter.FILTER_REJECT; }
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    while (w.nextNode()) parts.push(w.currentNode.nodeValue);
    const names = ['aria-label', 'title', 'aria-valuetext', 'data-tip', 'data-l'];
    const attrs = Array.from(el.querySelectorAll('[aria-label],[title],[aria-valuetext],[data-tip],[data-l]'))
      .map((e) => names.map((n) => e.getAttribute(n) || '').join(' | ')).join(' | ');
    return { text: parts.join(' | '), attrs, ticks };
  }
  function audit() {
    const res = [];
    const push = (state, el) => res.push(Object.assign({ state }, textOf(el)));
    push('page', doc.body);
    for (const p of Object.keys(D.machine.parts)) { S.part = p; renderMachine(); push('machine:' + p, $('#v-machine')); }
    for (const m of D.path.models) for (const g of Object.keys(m.configs)) { S.m2 = m.id; S.g2 = g; renderPath(); push(`path:${m.id}:${g}`, $('#v-path')); }
    for (const m of D.spec.models) { S.m3 = m.id; renderSpec(); push('spec:' + m.id, $('#v-spec')); }
    for (const m of D.conc.models) for (const p of Object.keys(m.sets)) for (let i = 0; i < D.conc.clients.v.length; i++) for (const x of ['clients', 'columns']) {
      Object.assign(S, { m4: m.id, p4: p, n4: i, x4: x }); renderConc(); push(`conc:${m.id}:${p}:${i}:${x}`, $('#v-conc'));
    }
    for (const m of D.ctx.models) for (let i = 0; i < D.ctx.stops.v.length; i++) { S.m5 = m.id; S.L5 = i; renderCtx(); push(`ctx:${m.id}:${i}`, $('#v-ctx')); }
    const pre = h('pre', { id: 'audit-out', hidden: true });
    pre.textContent = JSON.stringify(res);
    doc.body.append(pre);
  }

  // ------------------------------------------------------------------ boot
  const VIEWS = {
    machine: [buildMachine, renderMachine], path: [buildPath, renderPath], spec: [buildSpec, renderSpec], conc: [buildConc, renderConc], ctx: [buildCtx, renderCtx],
  };
  function boot(data) {
    D = data;
    readHash();
    bindStatic();
    renderReadout();
    for (const [k, [build, render]] of Object.entries(VIEWS)) {
      const v = $(`[data-view="${k}"]`);
      if (!v) continue;
      build(v);
      render();
    }
    renderMemory($('[data-view="memory"]'));
    renderLlama($('[data-view="llama"]'));
    renderLimits($('[data-view="limits"]'));
    renderMethod($('[data-view="method"]'));
    renderDownloads($('[data-view="downloads"]'));
    if (VIEW && $('#' + VIEW)) { doc.body.classList.add('solo'); $('#' + VIEW).classList.add('solo-on'); }
    if (AT && $('#' + AT)) $('#' + AT).scrollIntoView();
    if (AUDIT) audit();
    let lastW = innerWidth, t = 0;
    addEventListener('resize', () => {
      clearTimeout(t);
      t = setTimeout(() => { if (Math.abs(innerWidth - lastW) > 20) { lastW = innerWidth; renderConc(); renderCtx(); } }, 150);
    });
  }
  load().then(boot).catch((e) => {
    const p = h('p', { class: 'noscript' }, 'The data of this page could not be loaded (' + e + '). The same data is in data.json and in the PDF report.');
    doc.body.prepend(p);
  });
})();
