"""
PDF export for regulatory document drafts.
Produces an ICH/FDA-style formatted document from the draft dict
returned by draft_regulatory_document().
"""

from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Table, TableStyle, PageBreak,
)
from reportlab.platypus.flowables import KeepTogether


# ── Styles ──────────────────────────────────────────────────────────────────

def _build_styles():
    base = getSampleStyleSheet()

    title = ParagraphStyle(
        "AristonTitle",
        parent=base["Title"],
        fontSize=16,
        leading=20,
        spaceAfter=6,
        textColor=colors.HexColor("#1B3A5C"),
    )
    subtitle = ParagraphStyle(
        "AristonSubtitle",
        parent=base["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=4,
        textColor=colors.HexColor("#555555"),
    )
    h1 = ParagraphStyle(
        "AristonH1",
        parent=base["Heading1"],
        fontSize=12,
        leading=16,
        spaceBefore=14,
        spaceAfter=4,
        textColor=colors.HexColor("#1B3A5C"),
        borderPad=2,
    )
    body = ParagraphStyle(
        "AristonBody",
        parent=base["Normal"],
        fontSize=9,
        leading=13,
        spaceAfter=6,
    )
    disclaimer = ParagraphStyle(
        "Disclaimer",
        parent=base["Normal"],
        fontSize=7,
        leading=10,
        textColor=colors.HexColor("#888888"),
        spaceAfter=4,
    )
    return title, subtitle, h1, body, disclaimer


# ── Cover page ───────────────────────────────────────────────────────────────

def _cover(draft: dict, styles) -> list:
    title_style, subtitle_style, _, body_style, disclaimer_style = styles
    now = datetime.utcnow().strftime("%d %B %Y")

    elements = [
        Spacer(1, 0.5 * inch),
        Paragraph("ARISTON AI", ParagraphStyle(
            "Watermark", fontSize=9, textColor=colors.HexColor("#AAAAAA"),
            spaceAfter=2,
        )),
        Paragraph(draft["document_type"], title_style),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1B3A5C")),
        Spacer(1, 0.15 * inch),
        Paragraph(f"<b>Drug / Compound:</b> {draft['drug_name']}", body_style),
        Paragraph(f"<b>Indication:</b> {draft['indication']}", body_style),
    ]
    if draft.get("nct_id"):
        elements.append(Paragraph(f"<b>NCT Identifier:</b> {draft['nct_id']}", body_style))
    elements += [
        Paragraph(f"<b>Regulatory Framework:</b> {draft['guideline']}", body_style),
        Paragraph(f"<b>FDA Reference:</b> {draft['fda_reference']}", body_style),
        Spacer(1, 0.1 * inch),
        Paragraph(f"<b>Section(s) Drafted:</b> {draft['section_drafted'].title()}", body_style),
        Paragraph(f"<b>AI Model:</b> {draft.get('model', 'N/A')}", body_style),
        Paragraph(f"<b>Job ID:</b> {draft.get('job_id', 'N/A')}", body_style),
        Paragraph(f"<b>Generated:</b> {now} UTC", body_style),
        Spacer(1, 0.2 * inch),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")),
        Spacer(1, 0.1 * inch),
    ]

    # Grounding metadata table
    meta_rows = [
        ["Sources retrieved", str(draft.get("sources_used", 0))],
        ["ClinicalTrials.gov grounding", "Yes" if draft.get("trial_data_used") else "No"],
    ]
    table = Table(meta_rows, colWidths=[2.8 * inch, 3.5 * inch])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F5F5F5")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(
        "CONFIDENTIAL — DRAFT FOR REVIEW ONLY. "
        "This document was AI-assisted and requires expert regulatory review before submission. "
        "Sections marked [DATA REQUIRED] must be completed with verified study data.",
        disclaimer_style,
    ))
    elements.append(PageBreak())
    return elements


# ── Body ─────────────────────────────────────────────────────────────────────

def _body(draft_text: str, styles) -> list:
    _, _, h1_style, body_style, _ = styles
    elements = []

    for line in draft_text.splitlines():
        stripped = line.strip()
        if not stripped:
            elements.append(Spacer(1, 0.08 * inch))
            continue

        # Treat lines starting with a number+dot or "##" / "#" as section headers
        if (
            (stripped[:2].rstrip(".").isdigit() and stripped[1:3] in (". ", ". "))
            or stripped.startswith("##")
            or stripped.startswith("# ")
            or (len(stripped) > 2 and stripped[0].isdigit() and stripped[1] == ".")
        ):
            safe = stripped.lstrip("#").strip()
            elements.append(KeepTogether([
                Spacer(1, 0.04 * inch),
                Paragraph(safe, h1_style),
                HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#DDDDDD")),
            ]))
        else:
            # Escape XML special chars, preserve bold markers (**text**)
            safe = (
                stripped
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("**", "<b>", 1)
                .replace("**", "</b>", 1)
            )
            elements.append(Paragraph(safe, body_style))

    return elements


# ── Footer / page numbering ───────────────────────────────────────────────────

def _add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#888888"))
    page_text = f"Ariston AI — Confidential Draft  |  Page {doc.page}"
    canvas.drawCentredString(LETTER[0] / 2, 0.5 * inch, page_text)
    canvas.restoreState()


# ── Public API ────────────────────────────────────────────────────────────────

def render_pdf(draft: dict) -> bytes:
    """
    Convert a draft dict (from draft_regulatory_document) to PDF bytes.

    Args:
        draft: dict with keys: document_type, drug_name, indication, nct_id,
               guideline, fda_reference, section_drafted, draft (text), etc.

    Returns:
        Raw PDF bytes suitable for a FileResponse / StreamingResponse.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=0.75 * inch,
        title=draft["document_type"],
        author="Ariston AI",
        subject=f"{draft['drug_name']} — {draft['indication']}",
    )

    styles = _build_styles()
    story = _cover(draft, styles) + _body(draft.get("draft", ""), styles)

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    return buffer.getvalue()
