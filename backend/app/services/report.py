import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from ..config import UPLOAD_DIR

CAUSE_TEXT = {
    "human": "Likely human-related",
    "natural": "Likely natural / disaster-related",
    "unknown": "Unknown (insufficient evidence)",
}


def _img(name: str):
    try:
        return Image(str(UPLOAD_DIR / name), width=8 * cm, height=8 * cm,
                     kind="proportional")
    except Exception:
        return "Image unavailable"


def build_report(analysis) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            title=f"Change Detection Report {analysis.id}")
    s = getSampleStyleSheet()

    story = [
        Paragraph("Change Detection Report", s["Title"]),
        Paragraph(f"Analysis ID: {analysis.id} &nbsp;|&nbsp; "
                  f"Status: {analysis.status}", s["Normal"]),
        Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M}", s["Normal"]),
        Spacer(1, 14),
    ]

    pair = Table(
        [[_img(analysis.old_image_path), _img(analysis.new_image_path)],
         ["Old image", "New image"]],
        colWidths=[8.5 * cm, 8.5 * cm],
    )
    pair.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story += [pair, Spacer(1, 14), Paragraph("Detected changes", s["Heading2"])]

    rows = [["#", "Object", "Cause analysis", "Confidence"]]
    for i, r in enumerate(analysis.regions, 1):
        rows.append([str(i), r.object_type,
                     CAUSE_TEXT.get(r.cause_label, r.cause_label),
                     f"{round(r.confidence * 100)}%"])

    if len(rows) == 1:
        story.append(Paragraph("No changes detected.", s["Normal"]))
    else:
        t = Table(rows, colWidths=[1 * cm, 3.5 * cm, 8 * cm, 3 * cm])
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ]))
        story.append(t)

    story += [
        Spacer(1, 14),
        Paragraph(
            "Note: satellite images show what changed, not why. "
            "'Unknown' means there is not enough contextual evidence "
            "to determine the cause.", s["Italic"]),
    ]

    doc.build(story)
    return buf.getvalue()