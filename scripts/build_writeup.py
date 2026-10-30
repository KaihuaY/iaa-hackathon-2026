"""Render docs/writeup_final.md -> final_submission/writeup.docx (via a temporary HTML opened in Word COM).
Pass --pdf to also print a PDF with Edge headless (not part of the package).
Injects the office-premium table from reports/part2_office_premium.csv and the worked-example numbers, and counts PDF pages."""
import html, re, subprocess, sys, tempfile, time
from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\AI\Ad-hoc\IAA_AI"); FS = ROOT / "final_submission"
SRC = ROOT / "docs/writeup_final.md"; TMP = Path(tempfile.gettempdir()) / "writeup_render"; TMP.mkdir(exist_ok=True)
md = SRC.read_text(encoding="utf-8")

# ---- office table + example numbers from the generated CSV
op = pd.read_csv(ROOT / "reports/part2_office_premium.csv")
ex = op.set_index("row_id").loc["M00185311"]
md = md.replace("EX_RP", f"{ex.rp:,.2f}").replace("EX_P", f"{ex.office:,.2f}")
assert abs((ex.rp + ex["cat"] + ex.fixed) / ex.denominator - ex.office) < 0.01
log = pd.read_csv(ROOT / "experiments/log.csv")
md = md.replace("N_EXPERIMENTS", f"about {len(log) // 10 * 10}")
rows = ["| row_id | line | channel | exposure | SI / cover | RP | CAT | 1 − L | **P (row)** | P p.a. | written p.a. | ratio |", "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
for _, r in op.iterrows():
    si = r.cover_type if r.line == "motor" else f"{r.sum_insured:,.0f}"
    rows.append(f"| {r.row_id} | {r.line} | {r.channel} | {r.exposure:.4f} | {si} | {r.rp:,.2f} | {r["cat"]:,.2f} | {r.denominator:.2f} | **{r.office:,.2f}** | {r.office_pa:,.2f} | {r.written_premium:,.2f} | {r.ratio_to_written:.2f} |")
md = md.replace("OFFICE_TABLE", "\n".join(rows))
(TMP / "writeup_filled.md").write_text(md, encoding="utf-8")

# ---- minimal markdown -> html
def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
    return s
out, i, lines = [], 0, md.split("\n")
while i < len(lines):
    ln = lines[i]
    if ln.startswith("|"):
        tbl = []
        while i < len(lines) and lines[i].startswith("|"):
            tbl.append(lines[i]); i += 1
        hdr = [c.strip() for c in tbl[0].strip("|").split("|")]
        align = [("r" if c.strip().endswith(":") else "l") for c in tbl[1].strip("|").split("|")]
        out.append("<table><thead><tr>" + "".join(f"<th class='{a}'>{inline(c)}</th>" for c, a in zip(hdr, align)) + "</tr></thead><tbody>")
        for r in tbl[2:]:
            cells = [c.strip() for c in r.strip("|").split("|")]
            out.append("<tr>" + "".join(f"<td class='{a}'>{inline(c)}</td>" for c, a in zip(cells, align)) + "</tr>")
        out.append("</tbody></table>"); continue
    if ln.startswith("- "):
        out.append("<ul>")
        while i < len(lines) and lines[i].startswith("- "):
            out.append(f"<li>{inline(lines[i][2:])}</li>"); i += 1
        out.append("</ul>"); continue
    if re.match(r"\d+\. ", ln):
        out.append("<ol>")
        while i < len(lines) and re.match(r"\d+\. ", lines[i]):
            out.append(f"<li>{inline(re.sub(r'^\d+\. ', '', lines[i]))}</li>"); i += 1
        out.append("</ol>"); continue
    m = re.match(r"(#+) (.*)", ln)
    if m:
        out.append(f"<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>"); i += 1; continue
    if ln.strip():
        out.append(f"<p>{inline(ln)}</p>")
    i += 1
CSS = """
@page { size: A4; margin: 14mm 14mm 14mm 14mm; }
body { font-family: Calibri, 'Segoe UI', Arial, sans-serif; font-size: 9.3pt; line-height: 1.25; color: #111; max-width: 182mm; margin: 0 auto; }
h1 { font-size: 15pt; margin: 0 0 4pt 0; } h2 { font-size: 11.5pt; margin: 9pt 0 3pt 0; border-bottom: 1px solid #999; }
p { margin: 0 0 4pt 0; text-align: justify; } ul, ol { margin: 0 0 4pt 0; padding-left: 16pt; } li { margin: 0 0 1.5pt 0; }
table { border-collapse: collapse; width: 100%; margin: 3pt 0 5pt 0; font-size: 8pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 1.2pt 3pt; vertical-align: top; } th { background: #eee; text-align: left; }
td.r, th.r { text-align: right; } code { font-family: Consolas, monospace; font-size: 8.4pt; }
"""
doc = f"<!doctype html><html><head><meta charset='utf-8'><title>IAA Hackathon 2026 write-up</title><style>{CSS}</style></head><body>{''.join(out)}</body></html>"
(TMP / "writeup.html").write_text(doc, encoding="utf-8")

if '--pdf' in sys.argv:
    # ---- PDF via Edge headless
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    pdf = TMP / "writeup.pdf"
    tmp = TMP / "writeup_tmp.pdf"
    if tmp.exists(): tmp.unlink()
    subprocess.run([edge, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--user-data-dir=" + str(Path(tempfile.gettempdir()) / "edgeprof"),
                    f"--print-to-pdf={tmp}", (TMP / "writeup.html").as_uri()], check=True, timeout=120)
    for _ in range(20):
        if tmp.exists() and tmp.stat().st_size > 0: break
        time.sleep(0.5)
    try:
        tmp.replace(pdf)
    except PermissionError:
        pdf = TMP / f"writeup_{time.strftime('%H%M')}.pdf"; tmp.replace(pdf); print(f"WARNING: writeup.pdf is open in another program; wrote {pdf.name} instead (rename it when the viewer is closed)")
    b = pdf.read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", b))
    words = len(re.sub(r"\|.*", "", md).split())
    print(f"pdf {pdf.name}: {pdf.stat().st_size // 1024} KB, pages {pages}; words outside tables {words}")

# ---- DOCX via Word COM (open the HTML, save as docx)
ps = f"""
$w = New-Object -ComObject Word.Application; $w.Visible = $false
$d = $w.Documents.Open('{(TMP / "writeup.html")}')
$ps = $d.PageSetup; $ps.PaperSize = 7; $ps.TopMargin = 40; $ps.BottomMargin = 40; $ps.LeftMargin = 42; $ps.RightMargin = 42
'pages ' + $d.ComputeStatistics(2)
try {{ $d.SaveAs2('{(FS / "writeup.docx")}', 16); 'docx ok' }} catch {{ $d.SaveAs2('{(FS / "writeup_new.docx")}', 16); 'WARNING: writeup.docx locked; wrote writeup_new.docx' }}; $d.Close(); $w.Quit()
"""
r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=180)
print(r.stdout.strip() or r.stderr[-400:])
