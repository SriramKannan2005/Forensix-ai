"""
run_forensix.py — ForensiX AI command-line pipeline runner
-----------------------------------------------------------
Runs the full forensic pipeline on a single image:

    detection (ELA + noise + metadata + CNN)
        -> aggregator (risk profile)
        -> LLM reasoning (3 specialists, unless --skip-llm)
        -> Grad-CAM heatmap
        -> PDF report
        -> terminal summary

Usage:
    python run_forensix.py <image_path> [--skip-llm] [--no-heatmap] [--env dev|prod]
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252, which cannot encode the box-drawing
# characters used in the summary. Switch stdout/stderr to UTF-8 when possible.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def run(image_path: str, skip_llm: bool = False, save_heatmap: bool = True) -> dict:
    """Run the full pipeline and return the combined report dict."""
    from modules.image.detector import analyze_image
    from aggregator.aggregator import aggregate

    print(f"[1/5] Detection (ELA + noise + metadata + CNN): {Path(image_path).name}")
    detection = analyze_image(image_path, save_heatmap=save_heatmap)
    if detection.get("error"):
        raise RuntimeError(f"Detection failed: {detection['error']}")

    print("[2/5] Aggregating signals...")
    enriched = aggregate(detection)
    if not enriched["aggregator"]["aggregator_ready"]:
        errs = enriched["aggregator"]["validation_errors"]
        raise RuntimeError("Aggregator blocked: " + "; ".join(errs))

    report = {"detection": detection, "aggregator": enriched["aggregator"]}

    if not skip_llm:
        print("[3/5] Running LLM reasoning (3 specialists)...")
        try:
            from llm.reasoning_engine import run_llm_reasoning
            report.update(run_llm_reasoning(enriched))
        except Exception as e:  # noqa: BLE001 — LLM is optional
            print(f"      LLM reasoning skipped: {e}")
            report["llm_reasoning"] = {"final_verdict": "UNKNOWN",
                                       "overall_confidence": "NONE",
                                       "llm_error": str(e)}
    else:
        print("[3/5] LLM reasoning skipped (--skip-llm).")
        report["llm_reasoning"] = {"final_verdict": "DETECTION_ONLY",
                                   "overall_confidence": "NONE"}

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    print("[4/5] Generating Grad-CAM heatmap...")
    stem = Path(image_path).stem
    gradcam_overlay = None
    gradcam_path = None
    try:
        from modules.image.gradcam import generate_gradcam
        gradcam_result = generate_gradcam(
            image_input=image_path,
            output_path=str(ROOT / "outputs" / "gradcam" / f"{stem}_gradcam.png"),
        )
        gradcam_overlay = gradcam_result.get("overlay")
        gradcam_path = gradcam_result.get("output_path")
        if gradcam_result.get("error"):
            print(f"      Grad-CAM warning: {gradcam_result['error']}")
        report["gradcam"] = {
            "output_path": gradcam_path,
            "predicted_class": gradcam_result.get("predicted_class"),
            "predicted_score": gradcam_result.get("predicted_score"),
            "error": gradcam_result.get("error"),
        }
    except Exception as e:  # noqa: BLE001
        print(f"      Grad-CAM failed: {e}")

    # ── PDF report ────────────────────────────────────────────────────────────
    print("[5/5] Writing PDF report...")
    report_path = None
    try:
        from modules.reports.pdf_report import generate_report
        report_path = generate_report(
            image_path=image_path,
            forensic_result=report,
            gradcam_overlay=gradcam_overlay,
        )
        report["report_path"] = report_path
    except Exception as e:  # noqa: BLE001
        print(f"      PDF report failed: {e}")

    _print_summary(image_path, report, gradcam_path, report_path)
    return report


def _print_summary(image_path, report, gradcam_path, report_path):
    det = report.get("detection", {})
    agg = report.get("aggregator", {})
    llm = report.get("llm_reasoning", {})
    signals = det.get("signals", {})
    risk = agg.get("risk_profile", {})

    verdict = llm.get("final_verdict", "UNKNOWN")
    auth = det.get("authenticity_score", -1)
    cnn_score = signals.get("cnn_score")
    cnn_label = signals.get("cnn_label", "N/A")
    cnn_conf = signals.get("cnn_confidence", 0.0) or 0.0
    n_warn = risk.get("signals_in_warning", 0)
    n_crit = risk.get("signals_in_critical", 0)
    cnn_str = f"{cnn_score:.3f}" if isinstance(cnn_score, (int, float)) else "N/A"
    auth_str = f"{auth:.2f}" if isinstance(auth, (int, float)) else "N/A"

    print("\n" + "╔" + "═" * 46 + "╗")
    print("║       ForensiX AI — Analysis Result          ║")
    print("╠" + "═" * 46 + "╣")
    print(f"║ File:        {Path(image_path).name}")
    print(f"║ Verdict:     {verdict}")
    print(f"║ Auth Score:  {auth_str} / 1.00")
    print(f"║ CNN Score:   {cnn_str} ({cnn_label})")
    print(f"║ Confidence:  {cnn_conf:.1%}")
    print(f"║ ELA:         {signals.get('ela_suspicion', 'N/A')}")
    print(f"║ Flags:       {n_warn} warning, {n_crit} critical")
    print(f"║ GradCAM:     {gradcam_path or 'N/A'}")
    print(f"║ Report:      {report_path or 'N/A'}")
    print("╚" + "═" * 46 + "╝")


def main():
    parser = argparse.ArgumentParser(description="ForensiX AI CLI pipeline runner")
    parser.add_argument("image", help="Path to the image to analyse")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip the LLM reasoning stage (detection only)")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Do not save the ELA heatmap")
    parser.add_argument("--env", choices=["dev", "prod"], default=None,
                        help="LLM environment (sets FORENSIX_ENV)")
    args = parser.parse_args()

    if args.env:
        os.environ["FORENSIX_ENV"] = args.env

    if not Path(args.image).exists():
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    try:
        run(args.image, skip_llm=args.skip_llm, save_heatmap=not args.no_heatmap)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
