"""
pdf_report.py — PDF forensic report generator for ForensiX
-----------------------------------------------------------
Renders a full forensic analysis report to PDF using reportlab.

Public API:
    generate_report(image_path, forensic_result, output_path=None,
                    gradcam_overlay=None) -> str   # path to the written PDF

`forensic_result` is the combined pipeline dict:
    {
      "detection":   { ... ForensixResult ... },
      "aggregator":  { risk_profile, metadata_summary, auth_score_tier, ... },
      "llm_reasoning": { final_verdict, model_analyses, ... },   # optional
    }
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
)

ROOT = Path(__file__).resolve().parent.parent.parent


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(
        "FXTitle", parent=ss["Title"], fontSize=20, textColor=colors.HexColor("#0f3460"),
        spaceAfter=4))
    ss.add(ParagraphStyle(
        "FXSection", parent=ss["Heading2"], fontSize=13,
        textColor=colors.HexColor("#0f3460"), spaceBefore=10, spaceAfter=4))
    ss.add(ParagraphStyle(
        "FXBody", parent=ss["BodyText"], fontSize=9, leading=12))
    ss.add(ParagraphStyle(
        "FXSmall", parent=ss["BodyText"], fontSize=7.5, textColor=colors.grey))
    return ss


def _verdict_color(verdict: str) -> colors.Color:
    v = (verdict or "").upper()
    if v in ("AUTHENTIC",):
        return colors.HexColor("#00b894")
    if v in ("SUSPICIOUS", "UNKNOWN"):
        return colors.HexColor("#fdcb6e")
    return colors.HexColor("#d63031")


def _kv_table(rows, ss, col_widths=(55 * mm, 110 * mm)):
    data = [[Paragraph(str(k), ss["FXBody"]), Paragraph(str(v), ss["FXBody"])]
            for k, v in rows]
    t = Table(data, colWidths=list(col_widths))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f3f8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d6e0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def generate_report(
    image_path: str,
    forensic_result: dict,
    output_path: Optional[str] = None,
    gradcam_overlay: Optional[object] = None,
) -> str:
    """Render the forensic analysis report to a PDF and return its path."""
    ss = _styles()

    det = forensic_result.get("detection", {})
    agg = forensic_result.get("aggregator", {})
    llm = forensic_result.get("llm_reasoning", {})
    signals = det.get("signals", {})
    risk = agg.get("risk_profile", {})
    meta = signals.get("metadata", {})

    ts = datetime.now()
    name = Path(image_path).stem
    if output_path is None:
        out_dir = ROOT / "outputs" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"forensix_report_{ts:%Y%m%d_%H%M%S}_{name}.pdf")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    verdict = llm.get("final_verdict", "UNKNOWN")
    auth = det.get("authenticity_score", -1)

    story = []

    # ── 1. Header ─────────────────────────────────────────────────────────────
    story.append(Paragraph("ForensiX AI — Forensic Analysis Report", ss["FXTitle"]))
    story.append(Paragraph(
        f"Generated: {ts:%Y-%m-%d %H:%M:%S} &nbsp;|&nbsp; "
        f"Image: <b>{Path(image_path).name}</b>", ss["FXBody"]))
    vcol = _verdict_color(verdict)
    story.append(Spacer(1, 4))
    verdict_tbl = Table(
        [[Paragraph(f"<b>OVERALL VERDICT: {verdict}</b>",
                    ParagraphStyle("v", fontSize=13, textColor=colors.white))]],
        colWidths=[165 * mm])
    verdict_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), vcol),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(verdict_tbl)

    # ── 2. Executive Summary ──────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", ss["FXSection"]))
    story.append(_kv_table([
        ("Authenticity Score", f"{auth:.4f} / 1.00" if isinstance(auth, (int, float)) else auth),
        ("Auth Score Tier", agg.get("auth_score_tier", "N/A")),
        ("Final Verdict", verdict),
        ("Overall Confidence", llm.get("overall_confidence", "N/A")),
        ("Overall Risk", risk.get("overall_risk", "N/A")),
        ("Signals in Warning", risk.get("signals_in_warning", "N/A")),
        ("Signals in Critical", risk.get("signals_in_critical", "N/A")),
    ], ss))

    # ── 3. CNN Classification ─────────────────────────────────────────────────
    story.append(Paragraph("CNN Classification", ss["FXSection"]))
    story.append(_kv_table([
        ("CNN Score (P forged)", signals.get("cnn_score", "N/A")),
        ("CNN Label", signals.get("cnn_label", "N/A")),
        ("CNN Confidence", signals.get("cnn_confidence", "N/A")),
        ("Architecture", signals.get("cnn_arch", "N/A")),
        ("Validation Accuracy", signals.get("cnn_val_acc", "N/A")),
        ("Status", signals.get("cnn_note", "N/A")),
    ], ss))

    # ── 4. ELA Analysis ───────────────────────────────────────────────────────
    story.append(Paragraph("Error Level Analysis (ELA)", ss["FXSection"]))
    story.append(_kv_table([
        ("ELA Suspicion", f"{signals.get('ela_suspicion', 'N/A')}  [{risk.get('ela_suspicion_tier', 'N/A')}]"),
        ("ELA Mean Diff", f"{signals.get('ela_mean_diff', 'N/A')}  [{risk.get('ela_mean_tier', 'N/A')}]"),
        ("ELA Std Diff", signals.get("ela_std_diff", "N/A")),
        ("Regional Variance", f"{signals.get('ela_regional_variance', 'N/A')}  [{risk.get('ela_variance_tier', 'N/A')}]"),
        ("High-Energy Ratio", signals.get("ela_high_energy_ratio", "N/A")),
    ], ss))

    # ── 5. Metadata Analysis ──────────────────────────────────────────────────
    story.append(Paragraph("Metadata Analysis", ss["FXSection"]))
    story.append(_kv_table([
        ("File Format", meta.get("file_format", "N/A")),
        ("EXIF Present", meta.get("has_exif", "N/A")),
        ("Software Tag", meta.get("software_tag") or "NOT FOUND"),
        ("Camera Make", meta.get("camera_make") or "NOT FOUND"),
        ("Camera Model", meta.get("camera_model") or "NOT FOUND"),
        ("Metadata Flags", ", ".join(meta.get("metadata_flags", [])) or "NONE"),
    ], ss))

    # ── 6. Noise Analysis ─────────────────────────────────────────────────────
    story.append(Paragraph("Noise & Frequency Analysis", ss["FXSection"]))
    story.append(_kv_table([
        ("Noise Score", f"{signals.get('noise_score', 'N/A')}  [{risk.get('noise_tier', 'N/A')}]"),
        ("Block Noise CV", f"{signals.get('noise_block_cv', 'N/A')}  [{risk.get('noise_cv_tier', 'N/A')}]"),
        ("BG Min Corner Std", signals.get("bg_min_corner_std", "N/A")),
        ("Freq High Ratio", f"{signals.get('freq_high_ratio', 'N/A')}  [{risk.get('freq_tier', 'N/A')}]"),
        ("AI Smooth Flag", signals.get("ai_smooth_flag", "N/A")),
    ], ss))

    # ── 7. Active Flags ───────────────────────────────────────────────────────
    story.append(Paragraph("Active Flags", ss["FXSection"]))
    flags = det.get("flags", [])
    if flags:
        for f in flags:
            story.append(Paragraph(f"&bull; {f}", ss["FXBody"]))
    else:
        story.append(Paragraph("No flags raised.", ss["FXBody"]))

    # ── 8. GradCAM Heatmap ────────────────────────────────────────────────────
    if gradcam_overlay is not None:
        story.append(Paragraph("Grad-CAM Heatmap", ss["FXSection"]))
        try:
            buf = io.BytesIO()
            gradcam_overlay.save(buf, format="PNG")
            buf.seek(0)
            img = RLImage(ImageReader(buf))
            # Scale to max 120mm wide preserving aspect ratio
            iw, ih = gradcam_overlay.size
            max_w = 120 * mm
            scale = min(1.0, max_w / iw)
            img.drawWidth = iw * scale
            img.drawHeight = ih * scale
            story.append(img)
            story.append(Paragraph(
                "Regions in red indicate highest CNN suspicion.", ss["FXSmall"]))
        except Exception as e:  # noqa: BLE001
            story.append(Paragraph(f"Grad-CAM could not be embedded: {e}", ss["FXSmall"]))

    # ── 9. LLM Reasoning ──────────────────────────────────────────────────────
    analyses = llm.get("model_analyses", {})
    if analyses:
        story.append(Paragraph("LLM Reasoning (specialist excerpts)", ss["FXSection"]))
        for role, title in [
            ("visual_analyst", "Visual Anomaly Analyst"),
            ("metadata_analyst", "Metadata Timeline Analyst"),
            ("ai_pattern_analyst", "AI Pattern Analyst"),
        ]:
            data = analyses.get(role, {})
            if not data:
                continue
            reasoning = str(data.get("reasoning", "") or data.get("raw_response", ""))[:500]
            story.append(Paragraph(
                f"<b>{title}</b> — verdict: {data.get('verdict', 'N/A')} "
                f"(confidence: {data.get('confidence', 'N/A')})", ss["FXBody"]))
            if reasoning:
                story.append(Paragraph(reasoning, ss["FXBody"]))
            story.append(Spacer(1, 3))

    # ── 10. Footer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Generated by ForensiX AI | For research purposes only", ss["FXSmall"]))

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=18 * mm, bottomMargin=16 * mm,
        title="ForensiX AI — Forensic Analysis Report",
    )
    doc.build(story)
    return output_path
