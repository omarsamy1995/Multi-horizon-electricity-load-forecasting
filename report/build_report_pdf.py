"""
Automated Build Script for Generating the Academic Technical Report HTML & PDF.

Features:
- Converts report/Technical_Report.md into a self-contained, beautifully styled HTML document.
- Embeds high-resolution publication figures from experiments/.
- Renders professional IEEE/ACM-style academic CSS typography, print layout, page breaks, and tables.
- Invokes headless Chromium/Edge or WeasyPrint to export report/TeamName_Report.pdf.
"""

import os
import sys
import re
import subprocess
import shutil

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(REPORT_DIR)
MD_PATH = os.path.join(REPORT_DIR, "Technical_Report.md")
HTML_PATH = os.path.join(REPORT_DIR, "Technical_Report.html")
PDF_PATH = os.path.join(REPORT_DIR, "TeamName_Report.pdf")

def read_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def convert_markdown_tables_to_html(text: str) -> str:
    """Simple parser to convert Markdown tables to styled HTML tables."""
    lines = text.split("\n")
    in_table = False
    html_lines = []
    table_rows = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(stripped)
        else:
            if in_table:
                # Process the table
                in_table = False
                html_table = render_html_table(table_rows)
                html_lines.append(html_table)
            html_lines.append(line)

    if in_table:
        html_lines.append(render_html_table(table_rows))

    return "\n".join(html_lines)

def render_html_table(rows: list) -> str:
    if len(rows) < 2:
        return "\n".join(rows)

    header = [c.strip() for c in rows[0].strip("|").split("|")]
    # Check if second row is separator
    is_sep = all(re.match(r"^:?-+:?$", c.strip()) for c in rows[1].strip("|").split("|"))
    start_idx = 2 if is_sep else 1

    out = ['<div class="table-container"><table>', '<thead><tr>']
    for h in header:
        out.append(f"<th>{h}</th>")
    out.append('</tr></thead><tbody>')

    for r in rows[start_idx:]:
        cols = [c.strip() for c in r.strip("|").split("|")]
        out.append('<tr>')
        for c in cols:
            out.append(f"<td>{c}</td>")
        out.append('</tr>')

    out.append('</tbody></table></div>')
    return "\n".join(out)

def build_styled_html(md_content: str) -> str:
    # Convert tables first
    content = convert_markdown_tables_to_html(md_content)

    # Convert headers
    content = re.sub(r"^### (.*?)$", r"<h3>\1</h3>", content, flags=re.MULTILINE)
    content = re.sub(r"^## (.*?)$", r"<h2>\1</h2>", content, flags=re.MULTILINE)
    content = re.sub(r"^# (.*?)$", r"<h1>\1</h1>", content, flags=re.MULTILINE)

    # Convert bold and italics
    content = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", content)
    content = re.sub(r"\*(.*?)\*", r"<em>\1</em>", content)

    # Convert blockquotes
    content = re.sub(r"^> (.*?)$", r"<blockquote>\1</blockquote>", content, flags=re.MULTILINE)

    # Convert Markdown images
    # ![Caption](path)
    def img_replace(match):
        caption = match.group(1)
        path = match.group(2)
        # Normalize relative path to experiments/
        if "experiments" in path:
            fname = os.path.basename(path.replace("\\", "/"))
            rel_src = f"../experiments/{fname}"
            return f'<figure><img src="{rel_src}" alt="{caption}" /><figcaption>{caption}</figcaption></figure>'
        return f'<figure><img src="{path}" alt="{caption}" /><figcaption>{caption}</figcaption></figure>'

    content = re.sub(r"!\[(.*?)\]\((.*?)\)", img_replace, content)

    # Convert paragraphs
    paragraphs = content.split("\n\n")
    processed_p = []
    for p in paragraphs:
        p_str = p.strip()
        if not p_str:
            continue
        if p_str.startswith("<h1") or p_str.startswith("<h2") or p_str.startswith("<h3") or \
           p_str.startswith("<div") or p_str.startswith("<blockquote") or p_str.startswith("<figure") or \
           p_str.startswith("```"):
            processed_p.append(p_str)
        elif p_str.startswith("---"):
            processed_p.append('<hr class="section-divider" />')
        else:
            processed_p.append(f"<p>{p_str}</p>")

    body_html = "\n".join(processed_p)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Deep Learning Technical Report - Multi-Horizon Load Forecasting</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

  @page {{
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
    @bottom-center {{
      content: counter(page);
      font-family: 'Inter', sans-serif;
      font-size: 9pt;
      color: #64748b;
    }}
  }}

  body {{
    font-family: 'Crimson Pro', Georgia, serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #1e293b;
    background-color: #ffffff;
    max-width: 900px;
    margin: 0 auto;
    padding: 30px 25px;
  }}

  h1 {{
    font-family: 'Inter', sans-serif;
    font-size: 20pt;
    font-weight: 700;
    line-height: 1.25;
    color: #0f172a;
    text-align: center;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
  }}

  h2 {{
    font-family: 'Inter', sans-serif;
    font-size: 13pt;
    font-weight: 700;
    color: #1e3a8a;
    border-bottom: 1.5px solid #e2e8f0;
    padding-bottom: 4px;
    margin-top: 26px;
    margin-bottom: 10px;
    page-break-after: avoid;
  }}

  h3 {{
    font-family: 'Inter', sans-serif;
    font-size: 11pt;
    font-weight: 600;
    color: #334155;
    margin-top: 18px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}

  p {{
    margin-top: 0;
    margin-bottom: 10px;
    text-align: justify;
  }}

  blockquote {{
    border-left: 3.5px solid #3b82f6;
    background-color: #f8fafc;
    margin: 12px 0;
    padding: 8px 14px;
    font-style: italic;
    color: #334155;
  }}

  .table-container {{
    margin: 14px 0;
    overflow-x: auto;
    page-break-inside: avoid;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'Inter', sans-serif;
    font-size: 8.5pt;
    line-height: 1.35;
    margin: 6px 0;
  }}

  th, td {{
    border: 1px solid #cbd5e1;
    padding: 5.5px 7px;
    text-align: left;
  }}

  th {{
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 600;
  }}

  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}

  figure {{
    margin: 16px 0;
    text-align: center;
    page-break-inside: avoid;
  }}

  figure img {{
    max-width: 95%;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}

  figcaption {{
    font-family: 'Inter', sans-serif;
    font-size: 8.5pt;
    font-style: italic;
    color: #64748b;
    margin-top: 5px;
  }}

  hr.section-divider {{
    border: 0;
    height: 1px;
    background-color: #cbd5e1;
    margin: 22px 0;
  }}

  code {{
    font-family: 'Fira Code', Consolas, monospace;
    font-size: 9pt;
    background-color: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    color: #0f172a;
  }}

  .badge {{
    display: inline-block;
    padding: 2px 6px;
    font-family: 'Inter', sans-serif;
    font-size: 7.5pt;
    font-weight: 600;
    border-radius: 3px;
    background: #e2e8f0;
  }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""
    return html_template

def build_pdf_via_edge(html_path: str, pdf_path: str) -> bool:
    edge_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "msedge"
    ]
    edge_exe = None
    for cand in edge_candidates:
        if os.path.exists(cand) or shutil.which(cand):
            edge_exe = cand
            break

    if not edge_exe:
        print("[WARNING] msedge.exe not found on system path.")
        return False

    print(f"[INFO] Compiling PDF using headless Edge: {edge_exe}")
    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
            print(f"[SUCCESS] PDF generated successfully: {pdf_path} ({os.path.getsize(pdf_path)/(1024*1024):.2f} MB)")
            return True
        else:
            print(f"[ERROR] Edge run completed but PDF missing or empty: {res.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] Subprocess Edge failed: {e}")
        return False

def main():
    print("=" * 70)
    print("ACADEMIC TECHNICAL REPORT BUILD PIPELINE")
    print("=" * 70)

    if not os.path.exists(MD_PATH):
        print(f"[ERROR] Markdown file not found at: {MD_PATH}")
        sys.exit(1)

    print(f"[1/3] Reading source Markdown: {MD_PATH}")
    md_text = read_markdown(MD_PATH)

    print(f"[2/3] Generating styled academic HTML document: {HTML_PATH}")
    html_content = build_styled_html(md_text)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[SUCCESS] Wrote HTML to {HTML_PATH} ({len(html_content):,} chars)")

    print(f"[3/3] Exporting publication-grade PDF: {PDF_PATH}")
    pdf_built = build_pdf_via_edge(HTML_PATH, PDF_PATH)
    if not pdf_built:
        # Try fallback using python weasyprint if installed
        try:
            from weasyprint import HTML
            print("[INFO] Attempting fallback compilation with WeasyPrint...")
            HTML(HTML_PATH).write_pdf(PDF_PATH)
            print(f"[SUCCESS] WeasyPrint generated PDF: {PDF_PATH}")
            pdf_built = True
        except ImportError:
            print("[NOTE] You can also open report/Technical_Report.html in any browser and choose 'Print -> Save as PDF'.")

    print("=" * 70)
    print("BUILD COMPLETE!")
    print(f"Markdown : {MD_PATH}")
    print(f"HTML     : {HTML_PATH}")
    if os.path.exists(PDF_PATH):
        print(f"PDF      : {PDF_PATH} ({os.path.getsize(PDF_PATH)/(1024*1024):.2f} MB)")
    print("=" * 70)

if __name__ == "__main__":
    main()
