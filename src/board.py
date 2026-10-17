"""Render the decision board as self-contained HTML (no external resources).

Choices and notes autosave in the browser (localStorage) and can be downloaded as JSON / Markdown.
"""
import json

from decisions import DECISIONS

_CSS = """
.dboard{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--surface:#fcfcfb;--accent:#2a78d6;
  --hi:#d03b3b;--mh:#ec835a;--md:#eda100;--lo:#898781;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);max-width:1200px}
.dboard *{box-sizing:border-box}
.dboard .bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;position:sticky;top:0;z-index:5;background:var(--surface);
  padding:10px 0;border-bottom:1px solid var(--line);margin-bottom:12px}
.dboard .bar .prog{font-size:13px;color:var(--ink2);margin-right:auto}
.dboard button{font:inherit;font-size:13px;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:#fff;color:var(--ink);cursor:pointer}
.dboard button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.dboard button:hover{filter:brightness(.96)}
.dboard .card{border:1px solid var(--line);border-radius:10px;background:#fff;padding:16px 18px;margin:0 0 14px}
.dboard .card.done{border-left:4px solid #0ca30c}
.dboard .head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.dboard .rank{font-weight:700;color:var(--muted);font-variant-numeric:tabular-nums}
.dboard .title{font-weight:650;font-size:16px}
.dboard .badge{font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;color:#fff;white-space:nowrap}
.dboard .why{color:var(--ink2);font-size:13px;margin:4px 0 10px}
#dboard table{border-collapse:collapse;width:100%;font-size:13px;table-layout:fixed;margin:0}
#dboard col.c-pick{width:34px}#dboard col.c-lbl{width:30%}#dboard col.c-pro,#dboard col.c-con{width:auto}
#dboard th{text-align:left !important;color:var(--muted);font-weight:600;font-size:12px;border-bottom:1px solid var(--line);padding:6px 8px;background:transparent}
#dboard td{vertical-align:top;padding:8px;border-bottom:1px solid var(--line);text-align:left !important;background:transparent;white-space:normal;max-width:none}
#dboard tbody tr{background:transparent}
#dboard tr.opt{cursor:pointer}
#dboard tbody tr.opt:hover{background:#f6f8fc}
#dboard tbody tr.opt.sel{background:#eaf2fd}
#dboard td.lbl{font-weight:550}
.dboard .rec{display:inline-block;font-size:10.5px;font-weight:600;color:var(--accent);border:1px solid var(--accent);border-radius:4px;padding:0 5px;margin-left:6px}
.dboard .pro::before{content:"+ ";color:#006300;font-weight:700}
.dboard .con::before{content:"− ";color:#d03b3b;font-weight:700}
.dboard .suggest{font-size:13px;color:var(--ink2);margin:10px 0 6px}
.dboard .suggest b{color:var(--ink)}
.dboard textarea{width:100%;min-height:54px;font:inherit;font-size:13px;padding:8px;border:1px solid var(--line);border-radius:6px;resize:vertical}
.dboard .other{display:flex;gap:6px;align-items:center;margin-top:6px;font-size:13px;color:var(--ink2)}
.dboard .other input[type=text]{flex:1;font:inherit;font-size:13px;padding:5px 8px;border:1px solid var(--line);border-radius:6px}
.dboard .note{font-size:12px;color:var(--muted)}
@media (max-width:700px){#dboard table,#dboard tbody,#dboard tr,#dboard td{display:block}#dboard colgroup,#dboard thead{display:none}}
"""

_JS = r"""
(function(){
const D = __DATA__, KEY = "iaa-decision-board-v1", root = document.getElementById("dboard");
const colour = {"High":"var(--hi)","Medium-High":"var(--mh)","Medium":"var(--md)","Low-Medium":"var(--lo)","Low":"var(--lo)"};
let S = {}; try { S = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { S = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch(e) {} prog(); };
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const st = id => (S[id] = S[id] || {picks: [], other: "", notes: ""});

function card(d, i){
  const s = st(d.id), type = d.multi ? "checkbox" : "radio";
  const rows = d.options.map(o => `<tr class="opt${s.picks.includes(o.key)?" sel":""}" data-k="${o.key}">
    <td class="pick"><input type="${type}" name="d-${d.id}" value="${o.key}" ${s.picks.includes(o.key)?"checked":""} aria-label="${esc(o.label)}"></td>
    <td class="lbl">${o.key}. ${esc(o.label)}${o.rec?'<span class="rec">suggested</span>':""}</td>
    <td class="pro">${esc(o.pros)}</td><td class="con">${esc(o.cons)}</td></tr>`).join("");
  const done = s.picks.length || s.other.trim();
  return `<div class="card${done?" done":""}" id="c-${d.id}">
    <div class="head"><span class="rank">#${i+1}</span><span class="title">${esc(d.title)}</span>
      <span class="badge" style="background:${colour[d.impact]||"var(--accent)"}">${esc(d.impact)} impact</span></div>
    <div class="why">${esc(d.why)}</div>
    <table><colgroup><col class="c-pick"><col class="c-lbl"><col class="c-pro"><col class="c-con"></colgroup><thead><tr><th></th><th>Option</th><th>Pros</th><th>Cons</th></tr></thead><tbody>${rows}</tbody></table>
    <div class="other"><span>Other / my own option:</span><input type="text" data-f="other" value="${esc(s.other)}" placeholder="optional"></div>
    <div class="suggest"><b>Claude's suggestion:</b> ${esc(d.suggest)}</div>
    <textarea data-f="notes" placeholder="Your notes, questions or constraints for this decision…">${esc(s.notes)}</textarea>
  </div>`;
}

function prog(){
  const n = D.filter(d => { const s = st(d.id); return s.picks.length || s.other.trim(); }).length;
  root.querySelector(".prog").textContent = `${n} of ${D.length} decisions made · saved in this browser`;
  D.forEach(d => { const s = st(d.id), el = document.getElementById("c-"+d.id);
    if (el) el.classList.toggle("done", !!(s.picks.length || s.other.trim())); });
}

function exportObj(){
  return {exported: new Date().toISOString(), decisions: D.map((d,i) => { const s = st(d.id); return {
    rank: i+1, id: d.id, title: d.title, impact: d.impact,
    chosen: s.picks.map(k => k + ". " + d.options.find(o => o.key === k).label),
    other: s.other, notes: s.notes}; })};
}
function toMd(){
  return "# Modelling decisions\n\n" + exportObj().decisions.map(d =>
    `## ${d.rank}. ${d.title} (${d.impact})\n- Chosen: ${d.chosen.join("; ") || "—"}\n` +
    (d.other ? `- Other: ${d.other}\n` : "") + (d.notes ? `- Notes: ${d.notes.replace(/\n/g," ")}\n` : "")).join("\n");
}
function download(name, text, type){
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], {type}));
  a.download = name; document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
}

root.innerHTML = `<div class="bar"><span class="prog"></span>
  <button class="primary" data-a="json">Download choices (JSON)</button>
  <button data-a="md">Download as Markdown</button>
  <button data-a="sugg">Fill in suggestions</button>
  <button data-a="reset">Reset</button></div>` + D.map(card).join("") +
  `<p class="note">Choices autosave in this browser only. Download the JSON and drop it into the project's <code>docs/</code> folder so Claude can read it.</p>`;

root.addEventListener("click", e => {
  const a = e.target.closest("button[data-a]");
  if (a) {
    if (a.dataset.a === "json") download("decisions.json", JSON.stringify(exportObj(), null, 2), "application/json");
    if (a.dataset.a === "md") download("decisions.md", toMd(), "text/markdown");
    if (a.dataset.a === "sugg") { D.forEach(d => { const s = st(d.id); if (!s.picks.length) s.picks = d.options.filter(o => o.rec).map(o => o.key); }); save(); rerender(); }
    if (a.dataset.a === "reset" && confirm("Clear all choices and notes?")) { S = {}; save(); rerender(); }
    return;
  }
  const tr = e.target.closest("tr.opt"); if (!tr) return;
  const card = tr.closest(".card"), d = D.find(x => "c-"+x.id === card.id), s = st(d.id), k = tr.dataset.k;
  if (d.multi) s.picks = s.picks.includes(k) ? s.picks.filter(x => x !== k) : [...s.picks, k];
  else s.picks = s.picks.includes(k) && e.target.type !== "radio" ? [] : [k];
  card.querySelectorAll("tr.opt").forEach(r => { const on = s.picks.includes(r.dataset.k);
    r.classList.toggle("sel", on); r.querySelector("input").checked = on; });
  save();
});
root.addEventListener("input", e => {
  const f = e.target.dataset.f; if (!f) return;
  const d = D.find(x => "c-"+x.id === e.target.closest(".card").id); st(d.id)[f] = e.target.value; save();
});
function rerender(){ root.querySelectorAll(".card").forEach((el, i) => { el.outerHTML = card(D[i], i); }); prog(); }
prog();
})();
"""


def render_board(decisions=DECISIONS) -> str:
    return (f"<style>{_CSS}</style><div class='dboard' id='dboard'>Loading decision board…</div>"
            f"<script>{_JS.replace('__DATA__', json.dumps(decisions))}</script>")
