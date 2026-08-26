// Draws docs/data/sweep.json, which scripts/make_page_data.py reads out of
// results/. The sweep table and the committed generations are both the
// repository's own; nothing here is recomputed, and the two alphas that never
// finished are absent rather than filled in.

const el = (id) => document.getElementById(id);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const state = { data: null, alpha: 3.0, genAlpha: 3.0 };

// The four behavioural metrics, all percentages on one axis. Perplexity is not
// among them: it spans six orders of magnitude and would flatten everything
// else into the floor, which is exactly the mistake this page is about.
const SERIES = [
  { key: 'response_rate', label: 'benign response', dash: [], width: 2.4 },
  { key: 'coherent_response_rate', label: 'coherent', dash: [7, 4], width: 1.5 },
  { key: 'holdout_illicit_response_rate', label: 'illicit control', dash: [2, 3], width: 1.5 },
  { key: 'degenerate_rate', label: 'degenerate', dash: [10, 3, 2, 3], width: 1.5 },
];

function labelOnPaper(ctx, text, x, y, align = 'center') {
  const w = ctx.measureText(text).width;
  const left = align === 'center' ? x - w / 2 : align === 'right' ? x - w : x;
  const prev = ctx.fillStyle;
  ctx.fillStyle = css('--paper');
  ctx.fillRect(left - 3, y - 11, w + 6, 14);
  ctx.fillStyle = prev;
  ctx.textAlign = align;
  ctx.fillText(text, x, y);
}

function fitCanvas(canvas, h0) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w0 = canvas.clientWidth || 1200;
  canvas.width = Math.round(w0 * dpr);
  canvas.height = Math.round(h0 * dpr);
  canvas.style.height = h0 + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w0, h0);
  return { ctx, w: w0, h: h0 };
}

function draw() {
  const rows = state.data.sweep;
  const { ctx, w, h } = fitCanvas(el('plot'), 260);
  const pad = { l: 58, r: 132, t: 20, b: 48 };
  const iw = w - pad.l - pad.r;
  const ih = h - pad.t - pad.b;
  const X = (i) => pad.l + (i / (rows.length - 1)) * iw;
  const Y = (v) => pad.t + ih - (v / 100) * ih;

  ctx.strokeStyle = css('--hair');
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + ih); ctx.lineTo(pad.l + iw, pad.t + ih);
  ctx.stroke();
  ctx.font = "11px 'Courier New', monospace";
  ctx.textAlign = 'right';
  for (let v = 0; v <= 100; v += 25) {
    ctx.fillStyle = css('--faint');
    ctx.fillText(`${v}%`, pad.l - 8, Y(v) + 3);
    if (v) {
      ctx.strokeStyle = '#e8e3d6';
      ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(pad.l + iw, Y(v)); ctx.stroke();
    }
  }
  ctx.textAlign = 'center';
  rows.forEach((r, i) => {
    ctx.fillStyle = css('--faint');
    ctx.fillText(r.alpha.toFixed(1), X(i), pad.t + ih + 16);
  });
  ctx.fillStyle = css('--faint');
  ctx.fillText('intervention strength (alpha)', pad.l + iw / 2, h - 8);

  SERIES.forEach((s) => {
    ctx.save();
    ctx.setLineDash(s.dash);
    ctx.strokeStyle = css('--ox');
    ctx.lineWidth = s.width;
    ctx.beginPath();
    rows.forEach((r, i) => (i ? ctx.lineTo(X(i), Y(r[s.key])) : ctx.moveTo(X(i), Y(r[s.key]))));
    ctx.stroke();
    ctx.restore();
  });

  // Three of the four series end at 100%, so their labels land on the same
  // pixel. Push each one clear of the last rather than letting them overprint.
  const last = rows[rows.length - 1];
  const placed = SERIES.map((s) => ({ label: s.label, y: Y(last[s.key]) }))
    .sort((a, b) => a.y - b.y);
  placed.forEach((p, i) => {
    if (i && p.y - placed[i - 1].y < 15) p.y = placed[i - 1].y + 15;
  });
  ctx.textAlign = 'left';
  ctx.font = "12px 'Times New Roman', serif";
  ctx.fillStyle = css('--sub');
  placed.forEach((p) => ctx.fillText(p.label, pad.l + iw + 8, p.y + 4));

  // Where the model stopped producing language, marked on the axis the
  // behavioural metrics cannot show.
  const broken = rows.findIndex((r) => r.ppl_x_baseline > 100);
  if (broken >= 0) {
    ctx.save();
    ctx.strokeStyle = css('--bad');
    ctx.lineWidth = 1.4;
    ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(X(broken), pad.t); ctx.lineTo(X(broken), pad.t + ih); ctx.stroke();
    ctx.restore();
    ctx.font = "12px 'Times New Roman', serif";
    ctx.fillStyle = css('--bad');
    labelOnPaper(ctx, 'perplexity has exploded here', X(broken) - 6, pad.t + ih - 8, 'right');
  }

  const at = rows.findIndex((r) => r.alpha === state.alpha);
  if (at >= 0) {
    ctx.save();
    ctx.strokeStyle = css('--ink');
    ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(X(at), pad.t); ctx.lineTo(X(at), pad.t + ih); ctx.stroke();
    ctx.restore();
    SERIES.forEach((s) => {
      ctx.beginPath();
      ctx.arc(X(at), Y(rows[at][s.key]), 3.5, 0, Math.PI * 2);
      ctx.fillStyle = css('--ox');
      ctx.fill();
    });
  }
}

const fmtPpl = (x) =>
  x >= 1000 ? `${Math.round(x).toLocaleString('en-US')}x` : `${x.toFixed(2)}x`;

function render() {
  const r = state.data.sweep.find((x) => x.alpha === state.alpha);
  if (!r) return;
  el('r-resp').textContent = `${r.response_rate.toFixed(0)}%`;
  el('r-coh').textContent = `${r.coherent_response_rate.toFixed(0)}%`;
  el('r-deg').textContent = `${r.degenerate_rate.toFixed(0)}%`;
  el('r-ill').textContent = `${r.holdout_illicit_response_rate.toFixed(1)}%`;
  el('r-ppl').innerHTML =
    `<span class="ppl${r.ppl_x_baseline > 100 ? ' broken' : ''}"><b>${fmtPpl(r.ppl_x_baseline)}</b>` +
    `<small> baseline</small></span>`;
  const n = state.data.note;
  el('cap-what').textContent = `${n.model}, ${n.n_benign} benign prompts, ${n.n_holdout_illicit} held out`;
  el('cap-run').textContent = n.decoding;
  draw();

  const b = el('banner');
  if (r.ppl_x_baseline > 100) {
    b.className = 'banner alarm';
    b.textContent =
      `Every behavioural metric reads perfect at alpha ${r.alpha.toFixed(1)}: ${r.response_rate.toFixed(0)}% response, ` +
      `${r.degenerate_rate.toFixed(0)}% degenerate, ${r.coherent_response_rate.toFixed(0)}% coherent. ` +
      `Perplexity is ${fmtPpl(r.ppl_x_baseline)} baseline and the output is not language.`;
  } else if (r.degenerate_rate > 0) {
    b.className = 'banner alarm';
    b.textContent =
      `${r.degenerate_rate.toFixed(0)}% of responses are degenerate here, and coherence has fallen to ` +
      `${r.coherent_response_rate.toFixed(0)}%. This is the setting where it starts to show.`;
  } else if (r.holdout_illicit_response_rate > 50) {
    b.className = 'banner';
    b.textContent =
      `The model still writes fine, and the held-out illicit control has gone from 37.5% to ` +
      `${r.holdout_illicit_response_rate.toFixed(1)}%. The direction is not specific to over-refusal.`;
  } else {
    b.className = 'banner calm';
    b.textContent =
      `Alpha ${r.alpha.toFixed(1)}: benign response ${r.response_rate.toFixed(0)}%, perplexity ${fmtPpl(r.ppl_x_baseline)} ` +
      `baseline. Nothing has moved much in either direction yet.`;
  }
}

function gens() {
  const rows = state.data.samples.filter((s) => s.alpha === state.genAlpha);
  el('gens').innerHTML = rows
    .map((s) => {
      // ANSWERED on output that is not language is the whole point, so the
      // label is marked wrong where perplexity says it cannot be right.
      const wrong = state.genAlpha >= 3.0 && s.label === 'ANSWERED';
      return (
        `<div class="gen"><div class="head">` +
        `<span><b>${s.id}</b></span><span>${s.category.replace(/_/g, ' ')}</span>` +
        `<span class="verdict${wrong ? ' wrong' : ''}">classifier said ${s.label}</span></div>` +
        `<pre>${s.response.replace(/</g, '&lt;')}</pre></div>`
      );
    })
    .join('');
  const r = state.data.sweep.find((x) => x.alpha === state.genAlpha);
  el('gen-banner').className = state.genAlpha >= 3.0 ? 'banner alarm' : 'banner';
  el('gen-banner').textContent =
    state.genAlpha >= 3.0
      ? `All 108 responses at alpha 3.0 contain the same two-character cycle, and the classifier ` +
        `labelled every one of them ANSWERED. Perplexity for this row is ${fmtPpl(r.ppl_x_baseline)} baseline.`
      : `At alpha 2.5 the failure is visible to a reader: ${r.degenerate_rate.toFixed(0)}% are caught as ` +
        `degenerate and coherence has dropped to ${r.coherent_response_rate.toFixed(0)}%.`;
}

function picker(node, items, current, onPick) {
  node.innerHTML = '';
  items.forEach(({ key, label }) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.setAttribute('aria-pressed', String(key === current()));
    b.addEventListener('click', () => {
      onPick(key);
      [...node.children].forEach((c) => c.setAttribute('aria-pressed', String(c === b)));
    });
    node.appendChild(b);
  });
}

async function main() {
  const res = await fetch('./data/sweep.json');
  if (!res.ok) {
    el('banner').textContent = `Could not load the sweep (HTTP ${res.status}).`;
    return;
  }
  state.data = await res.json();

  picker(
    el('alphas'),
    state.data.sweep.map((r) => ({ key: r.alpha, label: r.alpha.toFixed(1) })),
    () => state.alpha,
    (k) => { state.alpha = k; render(); },
  );
  picker(
    el('genalpha'),
    [...new Set(state.data.samples.map((s) => s.alpha))].sort().map((a) => ({ key: a, label: a.toFixed(1) })),
    () => state.genAlpha,
    (k) => { state.genAlpha = k; gens(); },
  );
  window.addEventListener('resize', draw);

  render();
  gens();
}

main();
