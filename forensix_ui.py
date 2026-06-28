"""
forensix_ui.py — ForensiX AI Tkinter Desktop UI
-------------------------------------------------
Dark forensic terminal aesthetic. Features:
  • Drag-and-drop or browse image upload
  • Live animated progress bar across all 3 pipeline steps
  • Side-by-side original vs ELA heatmap viewer
  • Tabbed results: Verdict | Signals | Metadata | LLM Details
  • Color-coded verdict banner (green/yellow/red/magenta)
  • Per-signal severity indicators
  • Export JSON report button
  • Ollama model status checker

Run:
    python forensix_ui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import os
import sys
import time
from pathlib import Path
from PIL import Image, ImageTk

# ── Add project root and modules path to import search path ────────────────
ROOT = Path(__file__).resolve().parent
MODULES_ROOT = ROOT / "modules"
if MODULES_ROOT.exists():
    sys.path.insert(0, str(MODULES_ROOT))
sys.path.insert(0, str(ROOT))

# ── Color palette — dark forensic terminal ──────────────────────────────────
C = {
    "bg":         "#0D1117",   # deep near-black
    "bg2":        "#161B22",   # panel background
    "bg3":        "#1C2128",   # input / card
    "border":     "#30363D",   # subtle borders
    "text":       "#E6EDF3",   # primary text
    "text2":      "#8B949E",   # secondary text
    "text3":      "#484F58",   # muted text
    "accent":     "#58A6FF",   # blue accent (scan lines / highlights)
    "green":      "#3FB950",   # AUTHENTIC
    "yellow":     "#D29922",   # SUSPICIOUS
    "red":        "#F85149",   # TAMPERED
    "magenta":    "#BC8CFF",   # AI GENERATED
    "orange":     "#E3B341",   # WARNING flags
    "dim_green":  "#1A3A1F",
    "dim_yellow": "#3A2F0A",
    "dim_red":    "#3A1A1A",
    "dim_magenta":"#2A1F3A",
}

FONTS = {
    "title":   ("Courier New", 16, "bold"),
    "heading": ("Courier New", 11, "bold"),
    "mono":    ("Courier New", 10),
    "mono_sm": ("Courier New", 9),
    "label":   ("Segoe UI", 9),
    "label_b": ("Segoe UI", 9, "bold"),
    "big":     ("Courier New", 28, "bold"),
    "verdict": ("Courier New", 18, "bold"),
}

VERDICT_COLORS = {
    "AUTHENTIC":           (C["green"],   C["dim_green"]),
    "SUSPICIOUS":          (C["yellow"],  C["dim_yellow"]),
    "TAMPERED":            (C["red"],     C["dim_red"]),
    "AI_ASSISTED_TAMPER":  (C["red"],     C["dim_red"]),
    "FULLY_AI_GENERATED":  (C["magenta"], C["dim_magenta"]),
    "HUMAN_EDITED":        (C["red"],     C["dim_red"]),
    "ERROR":               (C["text3"],   C["bg3"]),
    "UNKNOWN":             (C["text3"],   C["bg3"]),
}

RISK_COLORS = {
    "NONE":     C["green"],
    "LOW":      C["text2"],
    "MEDIUM":   C["yellow"],
    "HIGH":     C["orange"],
    "CRITICAL": C["red"],
    "CLEAN":              C["green"],
    "LOW_SUSPICION":      C["text2"],
    "MODERATE_SUSPICION": C["yellow"],
    "HIGH_SUSPICION":     C["orange"],
    "CRITICAL":           C["red"],
    "NORMAL":             C["green"],
    "ELEVATED":           C["yellow"],
    "HIGH":               C["orange"],
    "VERY_HIGH":          C["red"],
    "EXTREME":            C["red"],
    "UNIFORM":            C["green"],
    "SLIGHTLY_UNEVEN":    C["text2"],
    "UNEVEN":             C["yellow"],
    "HIGHLY_UNEVEN":      C["red"],
    "CONSISTENT":         C["green"],
    "SLIGHTLY_INCONSISTENT": C["text2"],
    "INCONSISTENT":       C["yellow"],
    "HIGHLY_INCONSISTENT":C["red"],
    "UNAVAILABLE":        C["text3"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Reusable styled widgets
# ─────────────────────────────────────────────────────────────────────────────

def dark_frame(parent, **kw):
    return tk.Frame(parent, bg=kw.pop("bg", C["bg2"]), **kw)

def label(parent, text, font=None, color=None, **kw):
    return tk.Label(
        parent, text=text,
        font=font or FONTS["label"],
        fg=color or C["text"],
        bg=kw.pop("bg", C["bg2"]),
        **kw
    )

def separator(parent, color=C["border"], pady=4):
    f = tk.Frame(parent, bg=color, height=1)
    f.pack(fill="x", pady=pady)
    return f

def signal_row(parent, name: str, value, tier: str = "", bg=C["bg3"]):
    """Single signal row: name | value | tier badge"""
    row = tk.Frame(parent, bg=bg)
    row.pack(fill="x", padx=8, pady=1)

    tk.Label(row, text=f"  {name}", font=FONTS["mono_sm"],
             fg=C["text2"], bg=bg, width=28, anchor="w").pack(side="left")

    val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
    tk.Label(row, text=val_str, font=FONTS["mono_sm"],
             fg=C["text"], bg=bg, width=10, anchor="e").pack(side="left")

    if tier:
        tier_color = RISK_COLORS.get(tier, C["text3"])
        tk.Label(row, text=f"  [{tier}]", font=FONTS["mono_sm"],
                 fg=tier_color, bg=bg).pack(side="left")
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────

class ForensiXApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("ForensiX AI — Image Tampering Detector")
        self.geometry("1280x820")
        self.minsize(960, 700)
        self.configure(bg=C["bg"])
        self.resizable(True, True)

        # State
        self.image_path    = tk.StringVar()
        self.env_mode      = tk.StringVar(value="dev")
        self.skip_llm      = tk.BooleanVar(value=False)
        self.save_heatmap  = tk.BooleanVar(value=True)
        self.last_report   = None
        self._orig_photo   = None
        self._heat_photo   = None
        self._gradcam_photo = None
        self.last_gradcam_overlay = None
        self.last_gradcam_path    = None
        self.last_report_path     = None
        self._analysis_thread = None

        self._build_ui()
        self._style_ttk()

    # ── TTK styling ──────────────────────────────────────────────────────────

    def _style_ttk(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TNotebook",        background=C["bg"],  borderwidth=0)
        s.configure("TNotebook.Tab",    background=C["bg3"], foreground=C["text2"],
                    font=FONTS["mono_sm"], padding=[12, 5])
        s.map("TNotebook.Tab",
              background=[("selected", C["bg2"])],
              foreground=[("selected", C["accent"])])
        s.configure("Forensix.Horizontal.TProgressbar",
                    troughcolor=C["bg3"], background=C["accent"],
                    borderwidth=0, thickness=4)

    # ── UI Layout ────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top header
        self._build_header()
        # Main content: left panel + right panel
        content = tk.PanedWindow(self, orient="horizontal",
                                 bg=C["border"], sashwidth=2,
                                 sashrelief="flat")
        content.pack(fill="both", expand=True, padx=0, pady=0)

        left  = self._build_left_panel(content)
        right = self._build_right_panel(content)

        content.add(left,  minsize=360, width=420)
        content.add(right, minsize=500)

    def _build_header(self):
        hdr = tk.Frame(self, bg=C["bg"], height=54)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="⬡ FORENSIX AI",
                 font=FONTS["title"],
                 fg=C["accent"], bg=C["bg"]).pack(side="left", padx=20, pady=14)

        tk.Label(hdr, text="Image Tampering Detection System",
                 font=FONTS["mono_sm"],
                 fg=C["text3"], bg=C["bg"]).pack(side="left", padx=4, pady=14)

        # Env badge
        env_frame = tk.Frame(hdr, bg=C["bg"])
        env_frame.pack(side="right", padx=20)
        tk.Label(env_frame, text="MODE", font=FONTS["mono_sm"],
                 fg=C["text3"], bg=C["bg"]).pack(side="left")
        self._env_badge = tk.Label(env_frame, text=" DEV ",
                                   font=FONTS["mono_sm"],
                                   fg=C["bg"], bg=C["accent"],
                                   padx=4, pady=1)
        self._env_badge.pack(side="left", padx=6)

        # Divider
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

    # ── Left Panel ───────────────────────────────────────────────────────────

    def _build_left_panel(self, parent):
        panel = dark_frame(parent, bg=C["bg2"])

        # ── Upload section ───────────────────────────────────────────────────
        sec = dark_frame(panel)
        sec.pack(fill="x", padx=14, pady=(16, 0))

        label(sec, "TARGET IMAGE", font=FONTS["heading"],
              color=C["text3"], bg=C["bg2"]).pack(anchor="w")
        separator(sec, pady=4)

        # Drop zone
        self._drop_zone = tk.Label(
            sec,
            text="Click to browse\nor drag & drop image",
            font=FONTS["mono_sm"],
            fg=C["text3"], bg=C["bg3"],
            relief="flat", cursor="hand2",
            height=5
        )
        self._drop_zone.pack(fill="x", pady=6)
        self._drop_zone.bind("<Button-1>", lambda e: self._browse_image())

        # Path display
        self._path_label = tk.Label(
            sec, textvariable=self.image_path,
            font=FONTS["mono_sm"], fg=C["accent"],
            bg=C["bg2"], wraplength=360, justify="left"
        )
        self._path_label.pack(anchor="w", pady=2)

        # Image preview
        self._preview_frame = tk.Frame(sec, bg=C["bg3"], height=180)
        self._preview_frame.pack(fill="x", pady=6)
        self._preview_frame.pack_propagate(False)
        self._preview_label = tk.Label(self._preview_frame, bg=C["bg3"],
                                       text="No image loaded", font=FONTS["mono_sm"],
                                       fg=C["text3"])
        self._preview_label.pack(expand=True)

        # ── Options ──────────────────────────────────────────────────────────
        sep_frame = dark_frame(panel)
        sep_frame.pack(fill="x", padx=14, pady=(14, 0))
        label(sep_frame, "OPTIONS", font=FONTS["heading"],
              color=C["text3"], bg=C["bg2"]).pack(anchor="w")
        separator(sep_frame, pady=4)

        opt = dark_frame(panel)
        opt.pack(fill="x", padx=14)

        # Env toggle
        env_row = tk.Frame(opt, bg=C["bg2"])
        env_row.pack(fill="x", pady=3)
        tk.Label(env_row, text="Environment", font=FONTS["label"],
                 fg=C["text2"], bg=C["bg2"], width=16, anchor="w").pack(side="left")
        for val, txt in [("dev", "DEV (Ollama)"), ("prod", "PROD (API)")]:
            tk.Radiobutton(env_row, text=txt, variable=self.env_mode, value=val,
                           font=FONTS["label"], fg=C["text2"], bg=C["bg2"],
                           activebackground=C["bg2"], selectcolor=C["bg3"],
                           command=self._update_env_badge).pack(side="left", padx=6)

        # Checkboxes
        for var, txt in [(self.skip_llm, "Detection only (skip LLM)"),
                         (self.save_heatmap, "Save ELA heatmap")]:
            cb = tk.Checkbutton(opt, text=txt, variable=var,
                                font=FONTS["label"], fg=C["text2"], bg=C["bg2"],
                                activebackground=C["bg2"], selectcolor=C["bg3"])
            cb.pack(anchor="w", pady=1)

        # ── Ollama status ─────────────────────────────────────────────────────
        stat_frame = dark_frame(panel)
        stat_frame.pack(fill="x", padx=14, pady=(12, 0))
        label(stat_frame, "OLLAMA STATUS", font=FONTS["heading"],
              color=C["text3"], bg=C["bg2"]).pack(anchor="w")
        separator(stat_frame, pady=4)

        self._ollama_status = tk.Label(
            stat_frame, text="● Checking...",
            font=FONTS["mono_sm"], fg=C["text3"], bg=C["bg2"]
        )
        self._ollama_status.pack(anchor="w")

        tk.Button(stat_frame, text="Refresh Status",
                  font=FONTS["mono_sm"], fg=C["accent"], bg=C["bg3"],
                  relief="flat", cursor="hand2", bd=0, padx=8, pady=3,
                  command=self._check_ollama).pack(anchor="w", pady=4)

        # ── Outputs (report / gradcam launchers) ───────────────────────────────
        out_frame = dark_frame(panel)
        out_frame.pack(fill="x", padx=14, pady=(12, 0))
        label(out_frame, "OUTPUTS", font=FONTS["heading"],
              color=C["text3"], bg=C["bg2"]).pack(anchor="w")
        separator(out_frame, pady=4)

        out_btns = tk.Frame(out_frame, bg=C["bg2"])
        out_btns.pack(fill="x")
        self._report_btn = tk.Button(
            out_btns, text="Open Report", font=FONTS["mono_sm"],
            fg=C["text3"], bg=C["bg3"], relief="flat", cursor="hand2",
            bd=0, padx=8, pady=4, state="disabled", command=self._open_report)
        self._report_btn.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self._gradcam_btn = tk.Button(
            out_btns, text="View GradCAM", font=FONTS["mono_sm"],
            fg=C["text3"], bg=C["bg3"], relief="flat", cursor="hand2",
            bd=0, padx=8, pady=4, state="disabled", command=self._view_gradcam)
        self._gradcam_btn.pack(side="left", expand=True, fill="x", padx=(3, 0))

        # ── Analyse button ────────────────────────────────────────────────────
        btn_frame = tk.Frame(panel, bg=C["bg2"])
        btn_frame.pack(fill="x", padx=14, pady=16, side="bottom")

        self._analyse_btn = tk.Button(
            btn_frame,
            text="▶  RUN FORENSIC ANALYSIS",
            font=FONTS["heading"],
            fg=C["bg"], bg=C["accent"],
            relief="flat", cursor="hand2",
            bd=0, padx=16, pady=12,
            command=self._run_analysis
        )
        self._analyse_btn.pack(fill="x")

        # Progress bar
        self._progress = ttk.Progressbar(
            btn_frame, style="Forensix.Horizontal.TProgressbar",
            mode="indeterminate"
        )
        self._progress.pack(fill="x", pady=(6, 0))

        # Status line
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(btn_frame, textvariable=self._status_var,
                 font=FONTS["mono_sm"], fg=C["text3"],
                 bg=C["bg2"]).pack(anchor="w", pady=2)

        # Check Ollama on start
        threading.Thread(target=self._check_ollama, daemon=True).start()

        return panel

    # ── Right Panel ──────────────────────────────────────────────────────────

    def _build_right_panel(self, parent):
        panel = dark_frame(parent, bg=C["bg"])

        # Verdict banner
        self._verdict_frame = tk.Frame(panel, bg=C["bg3"], height=80)
        self._verdict_frame.pack(fill="x")
        self._verdict_frame.pack_propagate(False)

        self._verdict_label = tk.Label(
            self._verdict_frame,
            text="Awaiting Analysis",
            font=FONTS["verdict"],
            fg=C["text3"], bg=C["bg3"]
        )
        self._verdict_label.place(relx=0.5, rely=0.38, anchor="center")

        self._confidence_label = tk.Label(
            self._verdict_frame,
            text="",
            font=FONTS["mono_sm"],
            fg=C["text3"], bg=C["bg3"]
        )
        self._confidence_label.place(relx=0.5, rely=0.72, anchor="center")

        tk.Frame(panel, bg=C["border"], height=1).pack(fill="x")

        # Tabs
        self._tabs = ttk.Notebook(panel)
        self._tabs.pack(fill="both", expand=True)

        self._tab_verdict   = self._make_tab("VERDICT")
        self._tab_signals   = self._make_tab("SIGNALS")
        self._tab_heatmap   = self._make_tab("HEATMAP")
        self._tab_gradcam   = self._make_tab("GRADCAM")
        self._tab_metadata  = self._make_tab("METADATA")
        self._tab_llm       = self._make_tab("LLM DETAIL")

        self._tabs.add(self._tab_verdict,  text=" VERDICT ")
        self._tabs.add(self._tab_signals,  text=" SIGNALS ")
        self._tabs.add(self._tab_heatmap,  text=" HEATMAP ")
        self._tabs.add(self._tab_gradcam,  text=" GRADCAM ")
        self._tabs.add(self._tab_metadata, text=" METADATA ")
        self._tabs.add(self._tab_llm,      text=" LLM DETAIL ")

        self._build_verdict_tab()
        self._build_signals_tab()
        self._build_heatmap_tab()
        self._build_gradcam_tab()
        self._build_metadata_tab()
        self._build_llm_tab()

        return panel

    def _make_tab(self, name):
        f = tk.Frame(self._tabs, bg=C["bg2"])
        return f

    # ── Verdict Tab ──────────────────────────────────────────────────────────

    def _build_verdict_tab(self):
        t = self._tab_verdict
        t.columnconfigure(0, weight=1)

        self._verdict_body = tk.Frame(t, bg=C["bg2"])
        self._verdict_body.pack(fill="both", expand=True, padx=16, pady=12)

        self._v_placeholder = label(
            self._verdict_body,
            "Run analysis to see results.",
            color=C["text3"], bg=C["bg2"]
        )
        self._v_placeholder.pack(pady=40)

    def _populate_verdict_tab(self, report: dict):
        for w in self._verdict_body.winfo_children():
            w.destroy()

        llm = report.get("llm_reasoning", {})
        det = report.get("detection", {})
        agg = report.get("aggregator", {})

        verdict    = llm.get("final_verdict", "UNKNOWN")
        confidence = llm.get("overall_confidence", "UNKNOWN")
        auth_score = det.get("authenticity_score", -1)
        risk       = agg.get("risk_profile", {})
        overall_r  = risk.get("overall_risk", "UNKNOWN")
        tampering  = llm.get("tampering_report")

        # Auth score gauge
        score_frame = tk.Frame(self._verdict_body, bg=C["bg3"])
        score_frame.pack(fill="x", pady=(0, 10))

        tk.Label(score_frame, text="AUTHENTICITY SCORE",
                 font=FONTS["mono_sm"], fg=C["text3"], bg=C["bg3"]).pack(pady=(8, 2))

        score_color = C["green"] if auth_score > 0.7 else (
            C["yellow"] if auth_score > 0.4 else C["red"]
        )
        tk.Label(score_frame,
                 text=f"{auth_score:.4f}",
                 font=FONTS["big"],
                 fg=score_color, bg=C["bg3"]).pack()

        # Score bar
        bar_bg = tk.Frame(score_frame, bg=C["border"], height=6)
        bar_bg.pack(fill="x", padx=20, pady=(4, 10))
        bar_bg.update_idletasks()
        bar_w = max(1, int(bar_bg.winfo_width() * max(0, auth_score)))
        tk.Frame(bar_bg, bg=score_color, height=6,
                 width=bar_w).place(x=0, y=0, relwidth=auth_score)

        # Risk + confidence row
        info_row = tk.Frame(self._verdict_body, bg=C["bg2"])
        info_row.pack(fill="x", pady=4)

        for lbl, val, color in [
            ("RISK LEVEL",   overall_r, RISK_COLORS.get(overall_r, C["text"])),
            ("CONFIDENCE",   confidence, C["text"]),
            ("MODE",         report.get("aggregator", {}).get("auth_score_tier", "N/A"), C["text2"]),
        ]:
            box = tk.Frame(info_row, bg=C["bg3"])
            box.pack(side="left", expand=True, fill="x", padx=3, pady=2)
            tk.Label(box, text=lbl, font=FONTS["mono_sm"],
                     fg=C["text3"], bg=C["bg3"]).pack(pady=(6, 0))
            tk.Label(box, text=val, font=FONTS["heading"],
                     fg=color, bg=C["bg3"]).pack(pady=(0, 6))

        # CNN classifier result card
        signals = det.get("signals", {})
        if signals.get("cnn_model_loaded"):
            cnn_score = signals.get("cnn_score", 0.0) or 0.0
            cnn_label = signals.get("cnn_label", "UNKNOWN")
            cnn_conf  = signals.get("cnn_confidence", 0.0) or 0.0
            cnn_arch  = signals.get("cnn_arch", "?")
            cnn_acc   = signals.get("cnn_val_acc", 0.0) or 0.0
            cnn_color = C["red"] if cnn_label == "FORGED" else C["green"]

            cnn_card = tk.Frame(self._verdict_body, bg=C["bg3"])
            cnn_card.pack(fill="x", pady=(8, 4))
            tk.Label(cnn_card, text="CNN CLASSIFIER", font=FONTS["mono_sm"],
                     fg=C["text3"], bg=C["bg3"]).pack(anchor="w", padx=8, pady=(6, 0))
            tk.Label(cnn_card,
                     text=f"  Score: {cnn_score:.3f}  |  Label: {cnn_label}  "
                          f"|  Confidence: {cnn_conf:.1%}",
                     font=FONTS["heading"], fg=cnn_color, bg=C["bg3"]).pack(
                         anchor="w", padx=8)
            tk.Label(cnn_card,
                     text=f"  Model: {cnn_arch}  |  Val Acc: {cnn_acc:.2%}",
                     font=FONTS["mono_sm"], fg=C["text2"], bg=C["bg3"]).pack(
                         anchor="w", padx=8, pady=(0, 6))
        elif signals.get("cnn_note"):
            cnn_card = tk.Frame(self._verdict_body, bg=C["bg3"])
            cnn_card.pack(fill="x", pady=(8, 4))
            tk.Label(cnn_card, text=f"  CNN: {signals.get('cnn_note')}",
                     font=FONTS["mono_sm"], fg=C["text3"], bg=C["bg3"],
                     wraplength=560, justify="left").pack(anchor="w", padx=8, pady=6)

        separator(self._verdict_body, pady=6)

        # Active flags
        flags = det.get("flags", [])
        if flags:
            label(self._verdict_body, "ACTIVE FLAGS",
                  font=FONTS["heading"], color=C["text3"],
                  bg=C["bg2"]).pack(anchor="w")
            for f in flags:
                fc = C["red"] if "HIGH" in f or "CRITICAL" in f else (
                    C["yellow"] if "ELEVATED" in f else C["text2"]
                )
                tk.Label(self._verdict_body, text=f"  ⚑  {f}",
                         font=FONTS["mono_sm"], fg=fc, bg=C["bg2"],
                         anchor="w").pack(fill="x")
            separator(self._verdict_body, pady=6)

        # Tampering report
        if tampering:
            label(self._verdict_body, "TAMPERING REPORT",
                  font=FONTS["heading"], color=C["text3"],
                  bg=C["bg2"]).pack(anchor="w", pady=(4, 2))

            for field, key in [
                ("WHAT WAS CHANGED",  "what_was_changed"),
                ("WHEN WAS CHANGED",  "when_was_changed"),
                ("HOW (HUMAN/AI)",    "how_was_changed"),
                ("TOOL / TECHNIQUE",  "tool_or_technique"),
                ("LIKELY TOOL",       "likely_tool_used"),
                ("SOPHISTICATION",    "tampering_sophistication"),
            ]:
                val = tampering.get(key, "N/A")
                if val and val != "N/A":
                    row = tk.Frame(self._verdict_body, bg=C["bg3"])
                    row.pack(fill="x", pady=1)
                    tk.Label(row, text=f"  {field}", font=FONTS["mono_sm"],
                             fg=C["text3"], bg=C["bg3"], width=20, anchor="w").pack(side="left")
                    tk.Label(row, text=str(val), font=FONTS["mono_sm"],
                             fg=C["text"], bg=C["bg3"], wraplength=500,
                             justify="left", anchor="w").pack(side="left", fill="x", expand=True)

            regions = tampering.get("where_was_changed", [])
            if regions:
                label(self._verdict_body, "TAMPERED REGIONS",
                      font=FONTS["heading"], color=C["text3"],
                      bg=C["bg2"]).pack(anchor="w", pady=(8, 2))
                for r in regions:
                    box = tk.Frame(self._verdict_body, bg=C["bg3"])
                    box.pack(fill="x", pady=2)
                    tk.Label(box, text=f"  ● {r.get('region','?')}",
                             font=FONTS["label_b"], fg=C["red"],
                             bg=C["bg3"]).pack(anchor="w")
                    tk.Label(box, text=f"    Type: {r.get('anomaly_type','?')}",
                             font=FONTS["mono_sm"], fg=C["text2"],
                             bg=C["bg3"]).pack(anchor="w")
                    tk.Label(box, text=f"    Evidence: {r.get('evidence','?')}",
                             font=FONTS["mono_sm"], fg=C["text3"],
                             bg=C["bg3"]).pack(anchor="w", pady=(0, 4))
        else:
            # Bug 5 fix: don't show "No tampering detected" when hard signals
            # say otherwise. Check whether ELA flags or a suspicious/forged
            # auth_score_tier are present and surface a warning instead.
            hard_flag_keywords = ("HIGH_ELA_SUSPICION", "MODERATE_ELA_SUSPICION",
                                  "CRITICAL", "HIGH_SUSPICION")
            has_hard_flag = any(kw in f for f in flags for kw in hard_flag_keywords)
            auth_tier     = report.get("aggregator", {}).get("auth_score_tier", "")
            tier_suspicious = auth_tier in ("LIKELY_FORGED", "SUSPICIOUS", "UNCERTAIN")

            if has_hard_flag or tier_suspicious:
                # Signal-based evidence of tampering exists even though the LLM
                # returned AUTHENTIC — surface a warning rather than a clean pass.
                warn_frame = tk.Frame(self._verdict_body, bg=C["dim_yellow"])
                warn_frame.pack(fill="x", pady=4)
                tk.Label(
                    warn_frame,
                    text="  ⚠  LLM returned AUTHENTIC but hard forensic signals "
                         "indicate possible tampering.",
                    font=FONTS["mono_sm"], fg=C["yellow"],
                    bg=C["dim_yellow"], wraplength=600, justify="left",
                    anchor="w"
                ).pack(fill="x", padx=8, pady=6)
                tk.Label(
                    warn_frame,
                    text="  Check the SIGNALS tab for ELA evidence. "
                         "Consider running with 'Detection only' ticked to see "
                         "the signal-only verdict.",
                    font=FONTS["mono_sm"], fg=C["text2"],
                    bg=C["dim_yellow"], wraplength=600, justify="left",
                    anchor="w"
                ).pack(fill="x", padx=8, pady=(0, 6))
            else:
                label(self._verdict_body, "No tampering detected.",
                      color=C["green"], bg=C["bg2"]).pack(anchor="w", pady=8)

        # Export button
        separator(self._verdict_body, pady=8)
        tk.Button(
            self._verdict_body,
            text="⬇  Export Full JSON Report",
            font=FONTS["mono_sm"], fg=C["accent"], bg=C["bg3"],
            relief="flat", cursor="hand2", bd=0, padx=10, pady=6,
            command=self._export_report
        ).pack(anchor="w")

    # ── Signals Tab ──────────────────────────────────────────────────────────

    def _build_signals_tab(self):
        t = self._tab_signals
        self._signals_scroll = _ScrollFrame(t, bg=C["bg2"])
        self._signals_scroll.pack(fill="both", expand=True)
        self._signals_inner = self._signals_scroll.inner

        label(self._signals_inner, "Run analysis to see signals.",
              color=C["text3"], bg=C["bg2"]).pack(pady=40)

    def _populate_signals_tab(self, report: dict):
        for w in self._signals_inner.winfo_children():
            w.destroy()

        det = report.get("detection", {})
        agg = report.get("aggregator", {})
        signals = det.get("signals", {})
        risk    = agg.get("risk_profile", {})

        def section(title):
            f = tk.Frame(self._signals_inner, bg=C["bg2"])
            f.pack(fill="x", padx=8, pady=(10, 2))
            tk.Label(f, text=title, font=FONTS["heading"],
                     fg=C["text3"], bg=C["bg2"]).pack(anchor="w")
            tk.Frame(f, bg=C["border"], height=1).pack(fill="x", pady=2)

        # ELA signals
        section("ERROR LEVEL ANALYSIS (ELA)")
        ela_tiers = {
            "ela_suspicion":          risk.get("ela_suspicion_tier", ""),
            "ela_mean_diff":          risk.get("ela_mean_tier", ""),
            "ela_std_diff":           "",
            "ela_regional_variance":  risk.get("ela_variance_tier", ""),
            "ela_high_energy_ratio":  "",
        }
        for k, tier in ela_tiers.items():
            val = signals.get(k, "N/A")
            signal_row(self._signals_inner, k, val, tier, bg=C["bg3"])

        # Noise signals
        section("NOISE TEXTURE ANALYSIS")
        signal_row(self._signals_inner, "noise_score",
                   signals.get("noise_score", "N/A"),
                   risk.get("noise_tier", ""), bg=C["bg3"])
        signal_row(self._signals_inner, "noise_block_cv",
                   signals.get("noise_block_cv", "N/A"), "", bg=C["bg3"])

        # Risk profile
        section("AGGREGATED RISK PROFILE")
        for k in ["overall_risk", "signals_in_warning", "signals_in_critical"]:
            val = risk.get(k, "N/A")
            color = RISK_COLORS.get(str(val), C["text"])
            row = tk.Frame(self._signals_inner, bg=C["bg3"])
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=f"  {k}", font=FONTS["mono_sm"],
                     fg=C["text2"], bg=C["bg3"], width=28, anchor="w").pack(side="left")
            tk.Label(row, text=str(val), font=FONTS["mono_sm"],
                     fg=color, bg=C["bg3"]).pack(side="left")

        # CNN classifier
        section("CNN CLASSIFIER")
        signal_row(self._signals_inner, "cnn_model_loaded",
                   signals.get("cnn_model_loaded", False), "", bg=C["bg3"])
        if signals.get("cnn_model_loaded"):
            signal_row(self._signals_inner, "cnn_score",
                       signals.get("cnn_score", "N/A"),
                       risk.get("cnn_tier", ""), bg=C["bg3"])
            signal_row(self._signals_inner, "cnn_label",
                       signals.get("cnn_label", "N/A"), "", bg=C["bg3"])
            signal_row(self._signals_inner, "cnn_confidence",
                       signals.get("cnn_confidence", "N/A"), "", bg=C["bg3"])
        note = signals.get("cnn_note", "")
        if note:
            tk.Label(self._signals_inner, text=f"  {note}",
                     font=FONTS["mono_sm"], fg=C["text3"],
                     bg=C["bg2"], wraplength=560, justify="left").pack(anchor="w", padx=8, pady=2)

        # Processing time
        section("PERFORMANCE")
        signal_row(self._signals_inner, "processing_time",
                   det.get("processing_time", 0), "seconds", bg=C["bg3"])
        llm = report.get("llm_reasoning", {})
        if llm:
            signal_row(self._signals_inner, "llm_processing_time",
                       llm.get("llm_processing_time", 0), "seconds", bg=C["bg3"])

    # ── Heatmap Tab ──────────────────────────────────────────────────────────

    def _build_heatmap_tab(self):
        t = self._tab_heatmap
        ctrl = tk.Frame(t, bg=C["bg2"])
        ctrl.pack(fill="x", padx=12, pady=8)

        label(ctrl, "ORIGINAL", font=FONTS["heading"],
              color=C["text3"], bg=C["bg2"]).pack(side="left", padx=(0, 20))
        label(ctrl, "ELA HEATMAP  (red = high compression inconsistency = suspicious)",
              font=FONTS["mono_sm"], color=C["text3"], bg=C["bg2"]).pack(side="left")

        img_row = tk.Frame(t, bg=C["bg3"])
        img_row.pack(fill="both", expand=True, padx=8, pady=4)

        self._orig_canvas = tk.Label(img_row, bg=C["bg3"],
                                     text="Original image will appear here",
                                     font=FONTS["mono_sm"], fg=C["text3"])
        self._orig_canvas.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        tk.Frame(img_row, bg=C["border"], width=1).pack(side="left", fill="y")

        self._heat_canvas = tk.Label(img_row, bg=C["bg3"],
                                     text="ELA heatmap will appear here",
                                     font=FONTS["mono_sm"], fg=C["text3"])
        self._heat_canvas.pack(side="left", fill="both", expand=True, padx=4, pady=4)

    def _populate_heatmap_tab(self, report: dict):
        det  = report.get("detection", {})
        path = self.image_path.get()
        hmap = det.get("signals", {}).get("heatmap_path")

        def load_img(path, label_widget, max_w=480, max_h=420):
            try:
                img = Image.open(path)
                img.thumbnail((max_w, max_h), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                label_widget.configure(image=photo, text="")
                return photo
            except Exception:
                label_widget.configure(text="Image load failed")
                return None

        if path and Path(path).exists():
            self._orig_photo = load_img(path, self._orig_canvas)

        if hmap and Path(hmap).exists():
            self._heat_photo = load_img(hmap, self._heat_canvas)
        else:
            self._heat_canvas.configure(
                text="Heatmap not saved\n(enable 'Save ELA heatmap' option)",
                image=""
            )

    # ── GradCAM Tab ──────────────────────────────────────────────────────────

    def _build_gradcam_tab(self):
        t = self._tab_gradcam
        ctrl = tk.Frame(t, bg=C["bg2"])
        ctrl.pack(fill="x", padx=12, pady=8)
        label(ctrl, "CNN ATTENTION MAP",
              font=FONTS["heading"], color=C["text3"], bg=C["bg2"]).pack(side="left")
        label(ctrl, "  red regions = highest CNN suspicion",
              font=FONTS["mono_sm"], color=C["text3"], bg=C["bg2"]).pack(side="left")

        body = tk.Frame(t, bg=C["bg3"])
        body.pack(fill="both", expand=True, padx=8, pady=4)
        self._gradcam_canvas = tk.Label(
            body, bg=C["bg3"],
            text="Grad-CAM overlay will appear here after analysis",
            font=FONTS["mono_sm"], fg=C["text3"])
        self._gradcam_canvas.pack(expand=True, fill="both", padx=4, pady=4)

        self._gradcam_save_btn = tk.Button(
            t, text="⬇  Save GradCAM to Desktop", font=FONTS["mono_sm"],
            fg=C["accent"], bg=C["bg3"], relief="flat", cursor="hand2",
            bd=0, padx=10, pady=6, state="disabled",
            command=self._save_gradcam_to_desktop)
        self._gradcam_save_btn.pack(anchor="w", padx=8, pady=6)

    def _populate_gradcam_tab(self, report: dict):
        gc = report.get("gradcam", {})
        path = gc.get("output_path") or self.last_gradcam_path
        err  = gc.get("error")

        if self.last_gradcam_overlay is not None:
            try:
                img = self.last_gradcam_overlay.copy()
                img.thumbnail((640, 560), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._gradcam_canvas.configure(image=photo, text="")
                self._gradcam_photo = photo
                self._gradcam_save_btn.configure(state="normal")
                return
            except Exception:
                pass

        if path and Path(path).exists():
            try:
                img = Image.open(path)
                img.thumbnail((640, 560), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._gradcam_canvas.configure(image=photo, text="")
                self._gradcam_photo = photo
                self._gradcam_save_btn.configure(state="normal")
                return
            except Exception:
                pass

        msg = f"Grad-CAM unavailable\n{err}" if err else "Grad-CAM not generated"
        self._gradcam_canvas.configure(image="", text=msg)
        self._gradcam_save_btn.configure(state="disabled")

    def _save_gradcam_to_desktop(self):
        if self.last_gradcam_overlay is None and not (
                self.last_gradcam_path and Path(self.last_gradcam_path).exists()):
            messagebox.showinfo("No GradCAM", "Run analysis first.")
            return
        desktop = Path.home() / "Desktop"
        desktop.mkdir(parents=True, exist_ok=True)
        stem = Path(self.image_path.get()).stem or "image"
        dest = desktop / f"forensix_gradcam_{stem}.png"
        try:
            if self.last_gradcam_overlay is not None:
                self.last_gradcam_overlay.save(str(dest))
            else:
                Image.open(self.last_gradcam_path).save(str(dest))
            messagebox.showinfo("Saved", f"GradCAM saved to:\n{dest}")
        except Exception as e:
            messagebox.showerror("Save Failed", str(e))

    # ── Metadata Tab ─────────────────────────────────────────────────────────

    def _build_metadata_tab(self):
        t = self._tab_metadata
        self._meta_scroll = _ScrollFrame(t, bg=C["bg2"])
        self._meta_scroll.pack(fill="both", expand=True)
        self._meta_inner = self._meta_scroll.inner

        label(self._meta_inner, "Run analysis to see metadata.",
              color=C["text3"], bg=C["bg2"]).pack(pady=40)

    def _populate_metadata_tab(self, report: dict):
        for w in self._meta_inner.winfo_children():
            w.destroy()

        det     = report.get("detection", {})
        agg     = report.get("aggregator", {})
        signals = det.get("signals", {})
        meta    = signals.get("metadata", {})
        meta_s  = agg.get("metadata_summary", {})

        def row(name, val, color=None):
            r = tk.Frame(self._meta_inner, bg=C["bg3"])
            r.pack(fill="x", padx=8, pady=1)
            tk.Label(r, text=f"  {name}", font=FONTS["mono_sm"],
                     fg=C["text2"], bg=C["bg3"], width=26, anchor="w").pack(side="left")
            tk.Label(r, text=str(val), font=FONTS["mono_sm"],
                     fg=color or C["text"], bg=C["bg3"],
                     wraplength=400, justify="left").pack(side="left", anchor="w")

        def sec(title):
            f = tk.Frame(self._meta_inner, bg=C["bg2"])
            f.pack(fill="x", padx=8, pady=(10, 2))
            tk.Label(f, text=title, font=FONTS["heading"],
                     fg=C["text3"], bg=C["bg2"]).pack(anchor="w")
            tk.Frame(f, bg=C["border"], height=1).pack(fill="x", pady=2)

        sec("AGGREGATED INTERPRETATION")
        row("Origin Guess",      meta_s.get("origin_guess", "UNKNOWN"))
        row("Editing Software",  "YES" if meta_s.get("editing_software_detected") else "NO",
            C["red"] if meta_s.get("editing_software_detected") else C["green"])

        sec("EXIF DATA")
        row("EXIF Present",   "YES" if meta.get("has_exif") else "NO",
            C["green"] if meta.get("has_exif") else C["yellow"])
        row("Software Tag",   meta.get("software_tag") or "NOT FOUND",
            C["red"] if meta.get("software_tag") else C["text3"])
        row("Camera Make",    meta.get("camera_make")  or "NOT FOUND")
        row("Camera Model",   meta.get("camera_model") or "NOT FOUND")

        meta_flags = meta.get("metadata_flags", [])
        if meta_flags:
            sec("METADATA FLAGS")
            for f in meta_flags:
                tk.Label(self._meta_inner, text=f"  ⚑  {f}",
                         font=FONTS["mono_sm"], fg=C["red"],
                         bg=C["bg2"]).pack(anchor="w", padx=8)

        # LLM metadata verdict
        llm = report.get("llm_reasoning", {})
        if llm:
            meta_llm = llm.get("model_analyses", {}).get("metadata_analyst", {})
            if meta_llm:
                sec("LLM METADATA ANALYST VERDICT")
                timeline = meta_llm.get("timeline", {})
                row("Verdict",        meta_llm.get("verdict", "N/A"))
                row("Confidence",     meta_llm.get("confidence", "N/A"))
                row("Origin",         timeline.get("origin", "N/A"))
                row("Tampering Time", timeline.get("tampering_time", "N/A"))
                row("Time Evidence",  timeline.get("time_evidence", "N/A"))
                sw = meta_llm.get("software_used", {})
                row("Software Found", sw.get("detected_software", "N/A"))
                row("SW Evidence",    sw.get("software_evidence", "N/A"))
                row("When Changed",   meta_llm.get("when_was_it_changed", "N/A"))

                incons = meta_llm.get("metadata_inconsistencies", [])
                if incons:
                    sec("METADATA INCONSISTENCIES")
                    for i in incons:
                        tk.Label(self._meta_inner, text=f"  • {i}",
                                 font=FONTS["mono_sm"], fg=C["yellow"],
                                 bg=C["bg2"], wraplength=560,
                                 justify="left").pack(anchor="w", padx=8, pady=1)

    # ── LLM Detail Tab ───────────────────────────────────────────────────────

    def _build_llm_tab(self):
        t = self._tab_llm
        self._llm_scroll = _ScrollFrame(t, bg=C["bg2"])
        self._llm_scroll.pack(fill="both", expand=True)
        self._llm_inner = self._llm_scroll.inner

        label(self._llm_inner, "Run analysis to see LLM reasoning.",
              color=C["text3"], bg=C["bg2"]).pack(pady=40)

    def _populate_llm_tab(self, report: dict):
        for w in self._llm_inner.winfo_children():
            w.destroy()

        llm = report.get("llm_reasoning", {})
        if not llm:
            label(self._llm_inner, "LLM was skipped (--skip-llm mode).",
                  color=C["text3"], bg=C["bg2"]).pack(pady=40)
            return

        models_used = llm.get("models_used", {})
        analyses    = llm.get("model_analyses", {})
        errors      = llm.get("errors") or {}

        def sec(title, color=C["text3"]):
            f = tk.Frame(self._llm_inner, bg=C["bg2"])
            f.pack(fill="x", padx=8, pady=(12, 2))
            tk.Label(f, text=title, font=FONTS["heading"],
                     fg=color, bg=C["bg2"]).pack(anchor="w")
            tk.Frame(f, bg=C["border"], height=1).pack(fill="x", pady=2)

        def text_block(text, color=C["text2"]):
            tk.Label(self._llm_inner, text=text,
                     font=FONTS["mono_sm"], fg=color, bg=C["bg2"],
                     wraplength=620, justify="left", anchor="w").pack(
                         fill="x", padx=16, pady=2)

        # Consensus
        sec("CONSENSUS VERDICT", C["accent"])
        for k, v in [
            ("Final Verdict",      llm.get("final_verdict")),
            ("Overall Confidence", llm.get("overall_confidence")),
            ("Weighted Severity",  llm.get("weighted_severity")),
            ("Verdict Votes",      llm.get("verdict_votes")),
        ]:
            row = tk.Frame(self._llm_inner, bg=C["bg3"])
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=f"  {k}", font=FONTS["mono_sm"],
                     fg=C["text2"], bg=C["bg3"], width=22, anchor="w").pack(side="left")
            tk.Label(row, text=str(v), font=FONTS["mono_sm"],
                     fg=C["text"], bg=C["bg3"]).pack(side="left")

        # Per-model detail
        ROLE_LABELS = {
            "visual_analyst":     ("VISUAL ANOMALY ANALYST",    "llava / Claude"),
            "metadata_analyst":   ("METADATA TIMELINE ANALYST", "Mistral / GPT-4o"),
            "ai_pattern_analyst": ("AI PATTERN ANALYST",        "Gemma / Gemini"),
        }
        ROLE_KEYS = {
            "visual_analyst":     ["verdict", "confidence", "what_was_changed",
                                   "tampering_sophistication", "reasoning", "limitations"],
            "metadata_analyst":   ["verdict", "confidence", "when_was_it_changed",
                                   "reasoning", "limitations"],
            "ai_pattern_analyst": ["verdict", "confidence", "human_vs_ai_edit",
                                   "reasoning", "recommended_next_analysis", "limitations"],
        }

        for role, (title, subtitle) in ROLE_LABELS.items():
            model_name = models_used.get(role, "unknown")
            err = errors.get(role)

            hdr = tk.Frame(self._llm_inner, bg=C["bg2"])
            hdr.pack(fill="x", padx=8, pady=(14, 2))
            tk.Label(hdr, text=title, font=FONTS["heading"],
                     fg=C["accent"] if not err else C["red"],
                     bg=C["bg2"]).pack(anchor="w")
            tk.Label(hdr, text=f"  {subtitle}  ·  model: {model_name}",
                     font=FONTS["mono_sm"], fg=C["text3"],
                     bg=C["bg2"]).pack(anchor="w")
            tk.Frame(self._llm_inner, bg=C["border"], height=1).pack(
                fill="x", padx=8, pady=2)

            if err:
                text_block(f"ERROR: {err}", C["red"])
                continue

            data = analyses.get(role, {})
            for key in ROLE_KEYS.get(role, []):
                val = data.get(key)
                if not val:
                    continue
                row = tk.Frame(self._llm_inner, bg=C["bg3"])
                row.pack(fill="x", padx=8, pady=1)
                tk.Label(row, text=f"  {key}", font=FONTS["mono_sm"],
                         fg=C["text3"], bg=C["bg3"],
                         width=26, anchor="w").pack(side="left")
                tk.Label(row, text=str(val), font=FONTS["mono_sm"],
                         fg=C["text"], bg=C["bg3"],
                         wraplength=480, justify="left",
                         anchor="w").pack(side="left", fill="x", expand=True)

            # AI artifacts list
            if role == "ai_pattern_analyst":
                artifacts = data.get("ai_artifacts_found", [])
                if artifacts:
                    sec("  AI ARTIFACTS FOUND", C["yellow"])
                    for a in artifacts:
                        text_block(f"  • {a}", C["yellow"])

        # Next steps
        nxt = llm.get("recommended_next_analysis")
        if nxt:
            sec("RECOMMENDED NEXT ANALYSIS", C["text3"])
            text_block(nxt)

    # ── Pipeline runner ──────────────────────────────────────────────────────

    def _run_analysis(self):
        path = self.image_path.get().strip()
        if not path or not Path(path).exists():
            messagebox.showerror("No Image", "Please select a valid image file first.")
            return

        if self._analysis_thread and self._analysis_thread.is_alive():
            return

        self._analyse_btn.configure(state="disabled", text="  Analysing…")
        self._progress.start(10)
        self._status_var.set("Step 1/3: Running detection...")

        os.environ["FORENSIX_ENV"] = self.env_mode.get()

        self._analysis_thread = threading.Thread(
            target=self._pipeline_worker,
            args=(path,),
            daemon=True
        )
        self._analysis_thread.start()

    def _pipeline_worker(self, path: str):
        try:
            # ── Step 1: Detection ────────────────────────────────────────────
            self._set_status("Step 1/3: Running ELA + noise + metadata...")
            from modules.image.detector import analyze_image
            detection = analyze_image(path, save_heatmap=self.save_heatmap.get())

            if detection.get("error"):
                self._on_error(f"Detection failed: {detection['error']}")
                return

            # ── Step 2: Aggregator ───────────────────────────────────────────
            self._set_status("Step 2/3: Aggregating signals...")
            from aggregator.aggregator import aggregate
            enriched = aggregate(detection)

            if not enriched["aggregator"]["aggregator_ready"]:
                errs = enriched["aggregator"]["validation_errors"]
                self._on_error("Aggregator blocked: " + "; ".join(errs))
                return

            report = {
                "detection":  detection,
                "aggregator": enriched["aggregator"],
            }

            # ── Step 3: LLM reasoning ────────────────────────────────────────
            if not self.skip_llm.get():
                self._set_status("Step 3/3: Running 3 LLMs in parallel...")
                from llm.reasoning_engine import run_llm_reasoning
                llm_result = run_llm_reasoning(enriched)
                report.update(llm_result)
            else:
                # Build minimal LLM-like result from detection signals alone.
                # Bug 4 fix: derive verdict from both auth_score AND active flags
                # so "Detection only" mode reflects hard signal evidence correctly.
                auth  = detection.get("authenticity_score", 0.5)
                flags = detection.get("flags", [])

                # Hard-flag escalation: if HIGH or MODERATE ELA suspicion is
                # raised, floor the verdict to at least SUSPICIOUS regardless of
                # the raw score (which is currently ELA-only and may be noisy).
                if "HIGH_ELA_SUSPICION" in flags:
                    if auth < 0.5:
                        v = "TAMPERED"
                    else:
                        v = "SUSPICIOUS"
                elif "MODERATE_ELA_SUSPICION" in flags:
                    v = "SUSPICIOUS"
                elif auth > 0.7:
                    v = "AUTHENTIC"
                elif auth > 0.4:
                    v = "SUSPICIOUS"
                else:
                    v = "TAMPERED"

                # Build a minimal tampering_report so the verdict tab renders
                # signal evidence rather than the blank "No tampering detected"
                # message when signals indicate a problem.
                skip_tampering = None
                if v != "AUTHENTIC":
                    skip_tampering = {
                        "what_was_changed":  "Determined by signal analysis only — LLM skipped.",
                        "when_was_changed":  "Unknown — metadata analyst skipped.",
                        "how_was_changed":   "Unknown — AI pattern analyst skipped.",
                        "tool_or_technique": "Unknown",
                        "likely_tool_used":  "Unknown",
                        "ai_artifacts":      [],
                        "metadata_clues":    [],
                        "tampering_sophistication": "Unknown",
                    }

                report["llm_reasoning"] = {
                    "final_verdict":      v,
                    "overall_confidence": "LOW",
                    "authenticity_score": auth,
                    "models_used":        {"env": "skipped"},
                    "tampering_report":   skip_tampering,
                    "model_summaries":    {},
                    "model_analyses":     {},
                }

            # ── Step 4: Grad-CAM heatmap ─────────────────────────────────────
            self._set_status("Generating GradCAM...")
            self.last_gradcam_overlay = None
            self.last_gradcam_path    = None
            try:
                from modules.image.gradcam import generate_gradcam
                stem = Path(path).stem
                gc = generate_gradcam(
                    image_input=path,
                    output_path=str(ROOT / "outputs" / "gradcam" / f"{stem}_gradcam.png"),
                )
                self.last_gradcam_overlay = gc.get("overlay")
                self.last_gradcam_path    = gc.get("output_path")
                report["gradcam"] = {
                    "output_path":     gc.get("output_path"),
                    "predicted_class": gc.get("predicted_class"),
                    "predicted_score": gc.get("predicted_score"),
                    "error":           gc.get("error"),
                }
            except Exception as e:
                report["gradcam"] = {"error": str(e)}

            # ── Step 5: PDF report ───────────────────────────────────────────
            self._set_status("Writing PDF report...")
            self.last_report_path = None
            try:
                from modules.reports.pdf_report import generate_report
                self.last_report_path = generate_report(
                    image_path=path,
                    forensic_result=report,
                    gradcam_overlay=self.last_gradcam_overlay,
                )
                report["report_path"] = self.last_report_path
            except Exception as e:
                report["report_error"] = str(e)

            self.last_report = report
            self.after(0, self._on_success, report)

        except Exception as e:
            import traceback
            self._on_error(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def _on_success(self, report: dict):
        self._progress.stop()
        self._analyse_btn.configure(state="normal", text="▶  RUN FORENSIC ANALYSIS")

        llm     = report.get("llm_reasoning", {})
        verdict = llm.get("final_verdict", "UNKNOWN")
        conf    = llm.get("overall_confidence", "")

        fg, bg  = VERDICT_COLORS.get(verdict, (C["text"], C["bg3"]))
        self._verdict_frame.configure(bg=bg)
        self._verdict_label.configure(text=f"  {verdict}  ", fg=fg, bg=bg)
        self._confidence_label.configure(
            text=f"Confidence: {conf}", fg=fg, bg=bg
        )

        self._status_var.set(f"Complete — {verdict}")
        self._populate_verdict_tab(report)
        self._populate_signals_tab(report)
        self._populate_heatmap_tab(report)
        self._populate_gradcam_tab(report)
        self._populate_metadata_tab(report)
        self._populate_llm_tab(report)

        # Enable output launchers if artifacts were produced
        self._report_btn.configure(
            state="normal" if self.last_report_path
            and Path(self.last_report_path).exists() else "disabled")
        self._gradcam_btn.configure(
            state="normal" if (self.last_gradcam_overlay is not None or
                               (self.last_gradcam_path
                                and Path(self.last_gradcam_path).exists()))
            else "disabled")

        self._tabs.select(0)

    def _on_error(self, msg: str):
        self.after(0, lambda: [
            self._progress.stop(),
            self._analyse_btn.configure(state="normal",
                                        text="▶  RUN FORENSIC ANALYSIS"),
            self._status_var.set("Error — see details"),
            messagebox.showerror("Analysis Error", msg),
        ])

    def _set_status(self, msg: str):
        self.after(0, lambda: self._status_var.set(msg))

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files",   "*.*"),
            ]
        )
        if path:
            self.image_path.set(path)
            self._drop_zone.configure(
                text=Path(path).name,
                fg=C["accent"]
            )
            self._load_preview(path)
            self.last_report = None
            self.last_gradcam_overlay = None
            self.last_gradcam_path    = None
            self.last_report_path     = None
            if hasattr(self, "_report_btn"):
                self._report_btn.configure(state="disabled")
                self._gradcam_btn.configure(state="disabled")

    def _load_preview(self, path: str):
        try:
            img = Image.open(path)
            img.thumbnail((360, 175), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._preview_label.configure(image=photo, text="")
            self._preview_label.image = photo
        except Exception:
            self._preview_label.configure(text="Preview unavailable")

    def _update_env_badge(self):
        env = self.env_mode.get()
        self._env_badge.configure(
            text=f" {env.upper()} ",
            bg=C["accent"] if env == "dev" else C["green"]
        )

    def _check_ollama(self):
        import requests
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                needed = ["llava", "mistral", "gemma"]
                found  = [n for n in needed if any(n in m for m in models)]
                missing = [n for n in needed if n not in [f for f in found]]
                if missing:
                    txt   = f"● Running — missing: {', '.join(missing)}"
                    color = C["yellow"]
                else:
                    txt   = f"● Running — all 3 models ready"
                    color = C["green"]
            else:
                txt, color = "● Running but no models pulled", C["yellow"]
        except Exception:
            txt   = "● Not running — start with: ollama serve"
            color = C["red"]

        self.after(0, lambda: self._ollama_status.configure(text=txt, fg=color))

    def _open_report(self):
        if not (self.last_report_path and Path(self.last_report_path).exists()):
            messagebox.showinfo("No Report", "Run analysis first to generate a PDF report.")
            return
        self._open_file(self.last_report_path)

    def _view_gradcam(self):
        path = self.last_gradcam_path
        if not (path and Path(path).exists()):
            messagebox.showinfo("No GradCAM", "Run analysis first to generate a GradCAM.")
            return
        self._open_file(path)

    def _open_file(self, path: str):
        """Open a file with the OS default application (cross-platform)."""
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", path])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Open Failed", str(e))

    def _export_report(self):
        if not self.last_report:
            messagebox.showinfo("No Report", "Run analysis first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=f"forensix_{Path(self.image_path.get()).stem}.json"
        )
        if path:
            with open(path, "w") as f:
                json.dump(self.last_report, f, indent=2, default=str)
            messagebox.showinfo("Exported", f"Report saved to:\n{path}")


# ─────────────────────────────────────────────────────────────────────────────
# Scrollable frame helper
# ─────────────────────────────────────────────────────────────────────────────

class _ScrollFrame(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        bg = kw.get("bg", "white")

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.vbar   = tk.Scrollbar(self, orient="vertical",
                                   command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win  = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_frame_configure(self, _):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self.canvas.itemconfig(self._win, width=e.width)

    def _on_mousewheel(self, e):
        self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ForensiXApp()
    app.mainloop()