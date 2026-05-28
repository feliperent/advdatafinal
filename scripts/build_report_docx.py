"""Build the IN014 final-project DOCX from report/advdatafinal_report.md.

Matches the style of /Users/renteeee/Desktop/Methods of DM/reporttask3.docx:
  - 2.5cm margins on every side, A4
  - Default font 12pt, justified body
  - Cover page: date centred, title centred bold, course centred bold
  - Index / Table of Contents block before the body
  - Section headings bold inline (no oversized fonts)
  - Figure placeholders rendered as italic captions plus a coloured
    [INSERT IMAGE HERE: path/filename.png] marker if the PNG is missing.
    Drop the PNG in report/figures/ and re-run to auto-embed.

Output: report/advdatafinal_report.docx
"""
from __future__ import annotations

import re
import subprocess
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

REPO = Path(__file__).resolve().parent.parent
MD_SRC = REPO / "report" / "advdatafinal_report.md"
OUT = REPO / "report" / "advdatafinal_report.docx"
FIGURES = REPO / "report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Title-page metadata.
TODAY = date.today().strftime("%d/%m/%Y")
TITLE = "advdatafinal"
SUBTITLE = "A medallion pipeline with retrieval-augmented generation for 5-day directional return prediction"
COURSE = "IN014 Advanced Data Processing and Analysis"
AUTHOR = "Felipe Rentería Zuleta"
INSTITUTION = "Universitat Ramon Llull, La Salle Barcelona, 2025-2026"

# Body styling.
FONT = "Times New Roman"
BODY_SIZE = Pt(12)
CAPTION_SIZE = Pt(11)
CODE_SIZE = Pt(10)

# Image markers to look for in markdown.
IMAGE_PLACEHOLDER_RE = re.compile(r"\[\s*INSERT FIGURE [\w\d.]+:\s*([^\]]+?)\s*\]")


def _apply_default_font(run, size: Pt = BODY_SIZE, *, bold: bool = False, italic: bool = False) -> None:
    run.font.name = FONT
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), FONT)
    rfonts.set(qn("w:hAnsi"), FONT)
    rfonts.set(qn("w:cs"), FONT)
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic


def _page_setup(doc: Document) -> None:
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)
        # Page numbers top-right
        header = section.header
        p = header.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = p.add_run()
        fc1 = OxmlElement("w:fldChar"); fc1.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = "PAGE"
        fc2 = OxmlElement("w:fldChar"); fc2.set(qn("w:fldCharType"), "end")
        run._r.append(fc1); run._r.append(instr); run._r.append(fc2)
        _apply_default_font(run, size=Pt(10))


def _add_cover(doc: Document) -> None:
    # Date, top centred.
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _apply_default_font(p.add_run(TODAY))

    # Vertical breathing room.
    for _ in range(10):
        doc.add_paragraph()

    # Title centred bold.
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _apply_default_font(p.add_run(TITLE), bold=True)

    # Subtitle centred.
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _apply_default_font(p.add_run(SUBTITLE))

    doc.add_paragraph()
    doc.add_paragraph()

    # Course centred bold.
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _apply_default_font(p.add_run(COURSE), bold=True)

    # Final Project label, lighter weight.
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _apply_default_font(p.add_run("Final Project"))

    for _ in range(8):
        doc.add_paragraph()

    # Author + institution.
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _apply_default_font(p.add_run(AUTHOR), bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _apply_default_font(p.add_run(INSTITUTION))

    # Page break before TOC.
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)


def _add_toc(doc: Document) -> None:
    # Index heading.
    p = doc.add_paragraph()
    _apply_default_font(p.add_run("Index"), bold=True)

    entries = [
        ("Section 1. Introduction", "3"),
        ("Section 2. Data Governance", "5"),
        ("Section 3. Data Stack", "7"),
        ("Section 4. Data Model", "9"),
        ("Section 5. Findings", "12"),
        ("Section 6. Reflection", "14"),
        ("Appendix A. Code repository", "16"),
        ("Appendix B. Databricks workspace mirror", "16"),
        ("Appendix C. Reproducibility", "16"),
    ]
    for label, page in entries:
        p = doc.add_paragraph()
        run = p.add_run(label)
        _apply_default_font(run)
        # Tab leader of dots.
        run2 = p.add_run("\t" + "." * 60 + "\t" + page)
        _apply_default_font(run2)

    # List of figures.
    doc.add_paragraph()
    p = doc.add_paragraph()
    _apply_default_font(p.add_run("Table of figures"), bold=True)
    figs = [
        ("Figure 4.1. Databricks DLT pipeline DAG (advdatafinal_dlt)", "10"),
        ("Figure 4.2. Databricks Job graph (advdatafinal-pipeline)", "11"),
    ]
    for label, page in figs:
        p = doc.add_paragraph()
        _apply_default_font(p.add_run(label))
        run2 = p.add_run("\t" + "." * 60 + "\t" + page)
        _apply_default_font(run2)

    # List of tables.
    doc.add_paragraph()
    p = doc.add_paragraph()
    _apply_default_font(p.add_run("Table of tables"), bold=True)
    tables = [
        ("Table 1.1. External data sources, formats, and loaded volumes", "4"),
        ("Table 4.1. Nine declarative data-quality constraints inside the DLT pipeline", "11"),
        ("Table 5.1. AUC and hit rate per rung over 10 walk-forward folds", "13"),
        ("Table 5.2. Backtest cumulative return per rung over the 2024-2025 window", "13"),
    ]
    for label, page in tables:
        p = doc.add_paragraph()
        _apply_default_font(p.add_run(label))
        run2 = p.add_run("\t" + "." * 60 + "\t" + page)
        _apply_default_font(run2)

    # Page break before body.
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)


def _strip_inline_md(text: str) -> str:
    """Drop bold/italic/code markdown markers from inline runs."""
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text


def _render_table(doc: Document, lines: list[str]) -> None:
    rows = []
    for line in lines:
        parts = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(parts)
    # rows[0] is header, rows[1] is the |---|---| separator
    headers = rows[0]
    data = [r for r in rows[2:] if r and not all(c.startswith("-") for c in r)]
    table = doc.add_table(rows=1 + len(data), cols=len(headers))
    table.style = "Light Grid Accent 1"
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        _apply_default_font(p.add_run(_strip_inline_md(h)), size=Pt(10), bold=True)
    for i, row in enumerate(data, start=1):
        for j, val in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            _apply_default_font(p.add_run(_strip_inline_md(val)), size=Pt(10))
    doc.add_paragraph()


def _render_body(doc: Document) -> None:
    md = MD_SRC.read_text()
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Markdown table block (line containing pipes followed by a |---|---| row).
        if stripped.startswith("|") and stripped.endswith("|") and (i + 1 < len(lines)) and re.match(r"^\|[\s:-]+\|", lines[i + 1].strip()):
            block_start = i
            while i < len(lines) and lines[i].strip().startswith("|"):
                i += 1
            _render_table(doc, lines[block_start:i])
            continue

        # Image placeholder lines.
        m = IMAGE_PLACEHOLDER_RE.search(stripped)
        if m:
            path = m.group(1).strip()
            png = REPO / path
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if png.exists():
                p.add_run().add_picture(str(png), width=Inches(6.0))
            else:
                run = p.add_run(f"[ INSERT IMAGE HERE: {path} ]")
                _apply_default_font(run, bold=True)
                run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
            i += 1
            continue

        # Markdown headings.
        # Top-level (#) -> section heading, 12pt bold, with a page break before
        # so each numbered section starts on a new page.
        if stripped.startswith("# "):
            p = doc.add_paragraph()
            p.add_run().add_break(WD_BREAK.PAGE)
            p = doc.add_paragraph()
            _apply_default_font(p.add_run(stripped[2:].strip()), bold=True)
            doc.add_paragraph()
            i += 1
            continue
        if stripped.startswith("## "):
            doc.add_paragraph()
            p = doc.add_paragraph()
            _apply_default_font(p.add_run(stripped[3:].strip()), bold=True)
            i += 1
            continue
        if stripped.startswith("### "):
            p = doc.add_paragraph()
            _apply_default_font(p.add_run(stripped[4:].strip()), italic=True)
            i += 1
            continue

        # Empty line -> blank paragraph.
        if not stripped:
            doc.add_paragraph()
            i += 1
            continue

        # Body paragraph (justified).
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _apply_default_font(p.add_run(_strip_inline_md(stripped)))
        i += 1


def main() -> None:
    doc = Document()
    _page_setup(doc)
    _add_cover(doc)
    _add_toc(doc)
    _render_body(doc)
    doc.save(str(OUT))
    print(f"\n[ok] wrote {OUT}")
    print(f"     figures dir: {FIGURES}")
    print("     drop PNGs into report/figures/ and re-run to auto-embed:")
    print("       - report/figures/databricks_dlt_dag.png")
    print("       - report/figures/databricks_job_graph.png")


if __name__ == "__main__":
    main()
