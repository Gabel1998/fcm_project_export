#!/usr/bin/env python3
"""Konvertér REPORT.md til en selvstændig HTML-fil til PDF-eksport.

Indlejrer billeder som base64-data-URIs så HTML'en kan printes til PDF
af en browser uden at billed-stierne brydes. Output er REPORT.html som
derefter åbnes i Chrome og printes via Cmd+P → "Save as PDF".

Kør:
    .venv/bin/python scripts/06_markdown_to_html.py
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "REPORT.md"
REPORT_HTML = ROOT / "REPORT.html"

CSS = """
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
    max-width: 800px;
    margin: 2cm auto;
    padding: 0 1cm;
    color: #222;
    line-height: 1.55;
  }
  h1 { font-size: 1.9em; border-bottom: 2px solid #333; padding-bottom: .3em; margin-top: 0; }
  h2 { font-size: 1.4em; margin-top: 2em; border-bottom: 1px solid #ccc; padding-bottom: .2em; }
  h3 { font-size: 1.1em; margin-top: 1.5em; }
  p, li { font-size: 11pt; }
  table { border-collapse: collapse; margin: 1em 0; width: 100%; }
  th, td { border: 1px solid #bbb; padding: 6px 10px; text-align: left; font-size: 10pt; }
  th { background: #f3f3f3; }
  img { max-width: 100%; height: auto; display: block; margin: .8em auto; border: 1px solid #ddd; }
  em { font-style: italic; }
  /* Figur-billedtekster: kun em der er eneste indhold i en p (matcher
     markdown-mønstret "*Figur N: tekst*" som står på egen linje).
     Inline-italic som *ord* i brødtekst rammes ikke. */
  p > em:only-child { color: #555; font-size: 10pt; display: block; text-align: center; margin-top: -.4em; margin-bottom: 1em; }
  code { background: #f5f5f5; padding: 1px 5px; border-radius: 3px; font-size: 10pt; }
  hr { border: none; border-top: 1px solid #ccc; margin: 2em 0; }
  strong { color: #111; }
  @media print {
    body { margin: 1.5cm; max-width: none; }
    h2 { page-break-after: avoid; }
    table { page-break-inside: avoid; }
    /* Bevidst INGEN page-break-inside: avoid på img — lader Chrome
       brække frit i stedet for at efterlade store tomme felter. */
  }
</style>
"""


def embed_images(html: str) -> str:
    """Erstat src="docs/screenshots/...":stier med base64-data-URIs.

    Markdown-output har img-tags med alt før src, så regex matcher kun
    src-attributten uafhængigt af attribut-rækkefølge.
    """
    def replace(match: re.Match[str]) -> str:
        src = match.group(1)
        img_path = ROOT / src
        if not img_path.exists():
            return match.group(0)
        mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
        data = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return f'src="data:{mime};base64,{data}"'

    return re.sub(r'src="(docs/screenshots/[^"]+)"', replace, html)


def main() -> None:
    md_text = REPORT_MD.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html_body = embed_images(html_body)

    html = f"""<!DOCTYPE html>
<html lang="da">
<head>
  <meta charset="utf-8">
  <title>FCM Landmark Detection — Eksamensaflevering</title>
  {CSS}
</head>
<body>
{html_body}
</body>
</html>
"""
    REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"Genereret: {REPORT_HTML.relative_to(ROOT)}")
    print("Næste skridt: åbn i Chrome og brug Cmd+P → 'Save as PDF' → gem som REPORT.pdf")


if __name__ == "__main__":
    main()
