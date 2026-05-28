"""Build APA-styled DOCX report from the markdown source plus a custom cover page.

Pipeline:
  1. pandoc converts report/advdatafinal_report.md -> report/_body.docx
  2. python-docx prepends a Vitalux/Business-Law style cover page
  3. python-docx inserts figure placeholders for the two DAG screenshots
  4. Final file saved at report/advdatafinal_report.docx
"""
from __future__ import annotations

import shutil
import subprocess
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt, RGBColor

REPO = Path(__file__).resolve().parent.parent
MD_SRC = REPO / "report" / "advdatafinal_report.md"
BODY_DOCX = REPO / "report" / "_body.docx"
OUT_DOCX = REPO / "report" / "advdatafinal_report.docx"
FIGURES_DIR = REPO / "report" / "figures"


def run_pandoc() -> None:
    """Convert markdown to a base DOCX."""
    BODY_DOCX.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "pandoc",
        str(MD_SRC),
        "-o", str(BODY_DOCX),
        "--from", "gfm",
        "--reference-doc", "/dev/null" if not (REPO / "report" / "_reference.docx").exists() else str(REPO / "report" / "_reference.docx"),
    ]
    # Strip the non-existent reference flag
    cmd = [c for c in cmd if c != "/dev/null"]
    if "/dev/null" in cmd:
        cmd.remove("/dev/null")
    # Simpler: just run pandoc without reference doc if missing
    if not (REPO / "report" / "_reference.docx").exists():
        cmd = ["pandoc", str(MD_SRC), "-o", str(BODY_DOCX), "--from", "gfm"]
    print(f"pandoc: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def set_page_margins(doc: Document) -> None:
    """APA 7th edition: 1 inch (2.54 cm) margins on every side."""
    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)


def add_page_numbers(doc: Document) -> None:
    """Top-right page number per APA."""
    for section in doc.sections:
        header = section.header
        p = header.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = p.add_run()
        fldChar1 = OxmlElement("w:fldChar")
        fldChar1.set(qn("w:fldCharType"), "begin")
        instrText = OxmlElement("w:instrText")
        instrText.set(qn("xml:space"), "preserve")
        instrText.text = "PAGE"
        fldChar2 = OxmlElement("w:fldChar")
        fldChar2.set(qn("w:fldCharType"), "end")
        run._r.append(fldChar1)
        run._r.append(instrText)
        run._r.append(fldChar2)
        run.font.name = "Times New Roman"
        run.font.size = Pt(11)


def set_body_font(doc: Document, font_name: str = "Times New Roman", size: int = 11) -> None:
    """Apply Times New Roman 11pt to all paragraphs and runs."""
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if not run.font.name:
                run.font.name = font_name
            if not run.font.size:
                run.font.size = Pt(size)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        if not run.font.name:
                            run.font.name = font_name
                        if not run.font.size:
                            run.font.size = Pt(10)


def build_cover(doc: Document) -> None:
    """Vitalux / Business-Law style cover at the top of the doc.

    Layout: small date top-left, large title centred mid-page,
    subtitle, course, author block, institution at bottom.
    Single-spaced. Page break after.
    """
    # Insert cover before existing content. Easiest: build a fresh cover doc
    # and concatenate. Here we just prepend paragraphs.

    # Date (top-left, small)
    p_date = doc.paragraphs[0].insert_paragraph_before(date.today().strftime("%d/%m/%Y"))
    p_date.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_date.runs[0].font.name = "Times New Roman"
    p_date.runs[0].font.size = Pt(11)

    # Vertical spacer
    for _ in range(8):
        p_date.insert_paragraph_before("")

    # Title
    title = p_date.insert_paragraph_before("")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("advdatafinal")
    run.font.name = "Times New Roman"
    run.font.size = Pt(28)
    run.bold = True

    # Subtitle
    subtitle = p_date.insert_paragraph_before("")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run(
        "A medallion pipeline with retrieval-augmented generation\n"
        "for 5-day directional return prediction"
    )
    run.font.name = "Times New Roman"
    run.font.size = Pt(16)
    run.italic = True

    # Small gap
    for _ in range(3):
        p_date.insert_paragraph_before("")

    # Course
    course = p_date.insert_paragraph_before("")
    course.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = course.add_run("IN014 Advanced Data Processing and Analysis")
    run.font.name = "Times New Roman"
    run.font.size = Pt(14)

    # Final project marker
    fp = p_date.insert_paragraph_before("")
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run("Final Project")
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)

    # Spacer
    for _ in range(8):
        p_date.insert_paragraph_before("")

    # Author block
    author = p_date.insert_paragraph_before("")
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = author.add_run("Felipe Rentería Zuleta")
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    run.bold = True

    # Institution
    inst = p_date.insert_paragraph_before("")
    inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = inst.add_run("Universitat Ramon Llull, La Salle Barcelona\n2025-2026 academic year")
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)

    # Page break before the body
    pbreak = p_date.insert_paragraph_before("")
    pbreak.add_run().add_break(WD_BREAK.PAGE)


def insert_figure_placeholders(doc: Document) -> None:
    """Find the §4.2 'Architecture diagram' heading and append two figure boxes.

    The two figures (DAG and Job graph) are PNG screenshots the user took.
    If they exist in report/figures/, embed them. Otherwise leave a labelled
    placeholder paragraph the user can drag the PNG into.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dag_png = FIGURES_DIR / "databricks_dlt_dag.png"
    job_png = FIGURES_DIR / "databricks_job_graph.png"

    # Find §4.2 heading
    target_idx = None
    for i, p in enumerate(doc.paragraphs):
        if "Architecture diagram" in p.text and (p.style.name.startswith("Heading") or "##" in p.text):
            target_idx = i
            break

    if target_idx is None:
        print("  Architecture-diagram heading not found, appending figures at end")
        anchor = doc.paragraphs[-1]
    else:
        # Insert after the heading
        anchor = doc.paragraphs[target_idx]

    # Figure 1: DLT DAG
    fig1_caption = anchor.insert_paragraph_before("")
    fig1_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fig1_caption.add_run(
        "Figure 4.1. Databricks DLT pipeline DAG (advdatafinal_dlt). "
        "8 raw streaming tables (left) flow into 4 datos_masked redacted "
        "streaming tables, the silver layer (prices_typed view, silver_prices_cleaned "
        "SCD-1, two materialised views for features and fundamentals), and the gold "
        "layer (4 dimensions plus fct_feature_panel_daily). Star-schema joins on "
        "dim_date, dim_company, and dim_sector are explicit so the lineage graph "
        "renders dimension-to-fact edges."
    )
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)
    run.italic = True

    fig1_holder = anchor.insert_paragraph_before("")
    fig1_holder.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if dag_png.exists():
        from docx.shared import Inches
        fig1_holder.add_run().add_picture(str(dag_png), width=Inches(6.0))
    else:
        run = fig1_holder.add_run(f"[ INSERT IMAGE HERE: {dag_png.name} ]")
        run.font.name = "Times New Roman"
        run.font.size = Pt(11)
        run.bold = True
        run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)

    anchor.insert_paragraph_before("")

    # Figure 2: ML Job graph
    fig2_caption = anchor.insert_paragraph_before("")
    fig2_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fig2_caption.add_run(
        "Figure 4.2. Databricks Job graph (advdatafinal-pipeline). "
        "Task 1 (data_pipeline) triggers the DLT pipeline above. Task 2 (ml_pipeline) "
        "runs the serverless notebook databricksstuff/mlpipeline.py and depends on "
        "task 1. The notebook scores news and press releases with FinBERT, chunks "
        "and embeds 10-K and 8-K filings with MiniLM, computes 5 PCA components per "
        "company, builds fct_feature_panel_daily_full, trains two XGBoost rungs over "
        "walk-forward folds, and writes the backtest. LEFT ANTI JOIN logic in the "
        "FinBERT and MiniLM cells gives incremental scoring (4.4 minutes on re-runs "
        "vs 53.7 minutes from scratch)."
    )
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)
    run.italic = True

    fig2_holder = anchor.insert_paragraph_before("")
    fig2_holder.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if job_png.exists():
        from docx.shared import Inches
        fig2_holder.add_run().add_picture(str(job_png), width=Inches(4.0))
    else:
        run = fig2_holder.add_run(f"[ INSERT IMAGE HERE: {job_png.name} ]")
        run.font.name = "Times New Roman"
        run.font.size = Pt(11)
        run.bold = True
        run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)

    anchor.insert_paragraph_before("")


def main() -> None:
    run_pandoc()
    doc = Document(str(BODY_DOCX))

    set_page_margins(doc)
    add_page_numbers(doc)
    build_cover(doc)
    insert_figure_placeholders(doc)
    set_body_font(doc)

    doc.save(str(OUT_DOCX))
    BODY_DOCX.unlink(missing_ok=True)
    print(f"\n✓ wrote {OUT_DOCX}")
    print(f"  figures dir: {FIGURES_DIR}")
    print(f"  expected PNGs (drop them in to auto-embed on next build):")
    print(f"    - {FIGURES_DIR / 'databricks_dlt_dag.png'}")
    print(f"    - {FIGURES_DIR / 'databricks_job_graph.png'}")


if __name__ == "__main__":
    main()
