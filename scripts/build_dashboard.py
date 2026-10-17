"""Execute an EDA notebook and export it as a self-contained, offline HTML dashboard.

Usage:  python scripts/build_dashboard.py [notebooks/01_eda.py]

Steps: jupytext .py -> .ipynb, execute, export HTML without code cells, then strip every
external <script> (require.js, MathJax, mermaid loader) and fail if any remote script is left.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=ROOT)


def strip_external(html: str) -> str:
    html = re.sub(r'<script[^>]*\ssrc="https?://[^"]*"[^>]*>\s*</script>', "", html)
    # nbconvert's mermaid loader: an inline module that imports from a CDN when diagrams exist.
    html = re.sub(r'<script type="module">(?:(?!</script>).)*mermaid(?:(?!</script>).)*</script>', "", html, flags=re.S)
    return html


def main(src: str = "notebooks/01_eda.py"):
    py = ROOT / src
    nb = py.with_suffix(".ipynb")
    out = ROOT / "reports" / f"{py.stem}_dashboard.html"
    run("jupytext", "--to", "ipynb", str(py), "-o", str(nb))
    run("nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=1800",
        "--ExecutePreprocessor.kernel_name=python3", str(nb))
    run("nbconvert", "--to", "html", "--no-input", "--output-dir", str(out.parent), "--output", out.stem, str(nb))
    html = strip_external(out.read_text(encoding="utf-8"))
    left = re.findall(r'<script[^>]*\ssrc="https?://[^"]*"', html) + re.findall(r'import\("https?://', html)
    if left:
        raise SystemExit(f"external scripts remain: {left[:3]}")
    if "plotly.js v" not in html:
        raise SystemExit("plotly.js bundle not embedded")
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB), no external scripts")


if __name__ == "__main__":
    main(*sys.argv[1:])
