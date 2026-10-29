"""Road Damage Detection dashboard (Gradio), Midnight Steel theme, one-screen layout.

Final model: YOLOv10s (SGD, lr0=0.005) trained on the Roboflow "road-damage" dataset.
Run:  python app.py            -> http://127.0.0.1:7860
      python app.py --share    -> also prints a temporary public link
"""
import os
import sys
import tempfile
from collections import Counter

import cv2
import gradio as gr
import pandas as pd
from ultralytics import YOLO

MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
model = YOLO(MODEL_PATH)
CLASS_NAMES = [model.names[i] for i in sorted(model.names)]
NAME_TO_ID = {v: k for k, v in model.names.items()}

COLUMNS = ["Damage type", "Confidence", "x1", "y1", "x2", "y2"]
COUNT_COLS = ["Damage type", "Count"]

# ----------------------------------------------------------------------------
# Experiment results (test set, 681 images) - copied from the project report
# ----------------------------------------------------------------------------
ROUND1 = pd.DataFrame(
    [
        ["YOLOv10s + SGD", 0.612, 0.384, 0.597, 0.595, 0.778, 16.5, 8.1, 96.7, 6.7],
        ["YOLO11s + SGD", 0.600, 0.374, 0.587, 0.591, 0.779, 19.2, 9.4, 93.0, 6.5],
        ["YOLOv8s + SGD", 0.592, 0.365, 0.590, 0.578, 0.775, 22.5, 11.1, 95.9, 5.7],
        ["YOLOv10s + AdamW", 0.600, 0.363, 0.615, 0.565, 0.777, 16.5, 8.1, 95.7, 6.6],
        ["YOLO11s + AdamW", 0.585, 0.363, 0.565, 0.584, 0.780, 19.2, 9.4, 89.5, 6.5],
        ["YOLOv8s + AdamW", 0.580, 0.358, 0.616, 0.555, 0.776, 22.5, 11.1, 88.1, 5.7],
    ],
    columns=["Run", "mAP50", "mAP50-95", "Precision", "Recall", "Mean IoU",
             "Size (MB)", "Params (M)", "GPU FPS", "CPU FPS"],
)
ROUND2 = pd.DataFrame(
    [
        ["Baseline (lr0 0.01)", 640, 40, 0.612, 0.384, 0.597, 0.595],
        ["80 epochs", 640, 80, 0.600, 0.376, 0.598, 0.599],
        ["Image size 800", 800, 40, 0.610, 0.377, 0.607, 0.577],
        ["lr0 = 0.005 (FINAL)", 640, 40, 0.621, 0.390, 0.600, 0.621],
    ],
    columns=["Run", "Image size", "Epochs", "mAP50", "mAP50-95", "Precision", "Recall"],
)
PER_CLASS = pd.DataFrame(
    [
        ["block", 0.889], ["crack", 0.859], ["pothole", 0.693],
        ["transverse", 0.657], ["alligator", 0.361], ["longitudinal", 0.268],
    ],
    columns=["Class", "AP50"],
)
MAP_CHART = ROUND1[["Run", "mAP50-95"]].copy()
MAP_CHART["Run"] = (MAP_CHART["Run"].str.replace("YOLOv", "v").str.replace("YOLO11s", "v11s")
                    .str.replace(" + ", " ", regex=False))

PALETTE = {  # one fixed colour per damage type (boxes, legend, chips)
    "alligator": "#fb7185", "block": "#38bdf8", "crack": "#fbbf24", "edge": "#a78bfa",
    "longitudinal": "#34d399", "pothole": "#f97316", "transverse": "#22d3ee",
}
FALLBACK = ["#fb7185", "#38bdf8", "#fbbf24", "#a78bfa", "#34d399", "#f97316", "#22d3ee"]


def color_of(name):
    if name in PALETTE:
        return PALETTE[name]
    return FALLBACK[NAME_TO_ID.get(name, 0) % len(FALLBACK)]


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def legend_html():
    chips = "".join(f'<span class="chip"><i style="background:{color_of(n)}"></i>{n}</span>' for n in CLASS_NAMES)
    return f'<div class="legend">{chips}</div>'


def draw_boxes(rgb, boxes):
    """Draw boxes + labels with the class palette. boxes: list of (name, conf, x1, y1, x2, y2)."""
    img = rgb.copy()
    h, w = img.shape[:2]
    t = max(2, round(min(h, w) / 300))
    fs = max(0.5, min(h, w) / 900)
    for name, cf, x1, y1, x2, y2 in boxes:
        col = hex_to_rgb(color_of(name))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, t)
        label = f"{name} {cf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t - 1))
        ty = y1 if y1 - th - bl - 4 > 0 else y1 + th + bl + 4
        cv2.rectangle(img, (x1, ty - th - bl - 4), (x1 + tw + 8, ty), col, -1)
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        cv2.putText(img, label, (x1 + 4, ty - bl - 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (20, 20, 20) if lum > 150 else (255, 255, 255), max(1, t - 1), cv2.LINE_AA)
    return img


CSS = """
:root { color-scheme: dark; }
html, body, .gradio-container {min-height: 100vh !important; background: #0b1220 !important; color: #e8eef8 !important;}
.gradio-container {
    max-width: 100% !important; padding: 18px 24px !important;
    --body-background-fill: #0b1220; --background-fill-primary: #111c2e;
    --background-fill-secondary: #17263d; --block-background-fill: #111c2e;
    --input-background-fill: #0e192b; --border-color-primary: #263852;
    --border-color-accent: #38bdf8; --body-text-color: #e8eef8;
    --block-label-text-color: #b8c8dd; --input-text-color: #e8eef8;
}
footer {display: none !important;}
#topbar {
    align-items: center !important; flex-wrap: nowrap !important; gap: 10px !important;
    margin-bottom: 14px; padding: 16px 20px; border-radius: 17px;
    background: linear-gradient(112deg, #111c2e 0%, #173456 58%, #164e63 100%) !important;
    border: 1px solid #2b5473; box-shadow: 0 8px 24px rgba(0,0,0,.22);
    position: relative; overflow: hidden;
}
#topbar::after {content: ''; position: absolute; left: 0; right: 0; bottom: 0; height: 3px;
    background: linear-gradient(90deg, #38bdf8, #f59e0b, #38bdf8);}
#topbar > div:first-child {flex: 1 1 auto !important;}
#title {display: flex; align-items: center; gap: 15px;}
#title .logo {width: 48px; height: 48px; border-radius: 14px; display: grid; place-items: center;
    background: linear-gradient(135deg, #0284c7, #0f766e); border: 1px solid rgba(255,255,255,.15);}
#title h1 {margin: 0; font-size: 1.4rem; line-height: 1.2; color: #f4f8fc; letter-spacing: -.015em; font-weight: 750;}
#title .badges {margin-top: 7px; display: flex; gap: 7px; flex-wrap: wrap;}
#title .badge {font-size: .68rem; padding: 4px 10px; border-radius: 999px; color: #dbeafe;
    background: rgba(255,255,255,.07); border: 1px solid rgba(220,238,248,.15);}
#topbar button {flex: 0 0 42px !important; min-width: 42px !important; width: 42px !important; height: 42px;
    font-size: 1.05rem; padding: 0 !important; border-radius: 12px !important;
    background: rgba(255,255,255,.07) !important; border: 1px solid rgba(235,245,255,.16) !important; color: #f4f8fc !important;}
#topbar button:hover {background: rgba(56,189,248,.18) !important; border-color: #38bdf8 !important;}
.kpis {display: grid; grid-template-columns: 1.15fr 1fr 1fr 1fr 1fr; gap: 12px; margin: 0 0 14px;}
.kpi {background: #111c2e; border: 1px solid #263852; border-radius: 14px; padding: 12px 14px 12px 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,.12); position: relative; overflow: hidden;}
.kpi::before {content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--accent, #38bdf8);}
.kpi .v {font-size: 1.12rem; font-weight: 720; line-height: 1.35; color: #f1f5f9; display: flex; align-items: center; gap: 8px;}
.kpi .l {font-size: .65rem; color: #9fb1c9; text-transform: uppercase; letter-spacing: .075em; font-weight: 650; margin-top: 3px;}
.kpi .dot {width: 9px; height: 9px; flex: 0 0 9px; border-radius: 50%; background: var(--accent);
    box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 16%, transparent);}
.workspace-panel {background: #111c2e !important; border: 1px solid #263852 !important; border-radius: 16px !important;
    padding: 14px !important; box-shadow: 0 5px 16px rgba(0,0,0,.12);}
.gradio-container .block, .gradio-container .form, .gradio-container .tabs,
.gradio-container [role="tabpanel"] {background-color: transparent;}
.gradio-container input, .gradio-container textarea, .gradio-container select {
    background-color: #0e192b !important; color: #e8eef8 !important; border-color: #30445f !important;}
.gradio-container label {color: #c5d3e5 !important;}
.gradio-container .prose, .gradio-container p {color: #d5e0ee;}
button.primary {background: linear-gradient(135deg, #0284c7, #0f766e) !important; color: #fff !important;
    font-weight: 700 !important; border: none !important; letter-spacing: .01em;
    box-shadow: 0 3px 9px rgba(2,132,199,.2);}
button.primary:hover {filter: brightness(1.08);}
button.secondary {font-weight: 600 !important;}
button[role=tab] {border-radius: 10px !important; color: #b8c8dd !important;}
button[role=tab][aria-selected=true] {color: #7dd3fc !important; border-color: #38bdf8 !important;
    font-weight: 700; background: #172f49 !important;}
.legend {display: flex; flex-wrap: wrap; gap: 7px; padding: 10px 2px 0;}
.chip {font-size: .72rem; padding: 4px 11px 4px 8px; border-radius: 999px; display: inline-flex; align-items: center;
    gap: 7px; border: 1px solid #30445f; background: #17263d; color: #e8eef8; font-weight: 600;}
.chip i {width: 9px; height: 9px; border-radius: 3px; display: inline-block;}
.overlay {position: fixed !important; top: 6vh; left: 50%; transform: translateX(-50%);
    width: min(1050px, 94vw); max-height: 88vh; overflow: auto; z-index: 1000;
    background: #111c2e !important; color: #e8eef8 !important; border: 1px solid #2b5473;
    border-radius: 18px; padding: 20px 24px !important;
    box-shadow: 0 0 0 100vmax rgba(3,8,15,.62), 0 24px 70px rgba(0,0,0,.4);}
.overlay h3 {margin: 0 0 8px 0; color: #f1f5f9;}
.note {font-size: .8rem; color: #9fb1c9;}
@media (max-width: 900px) {
    html, body, .gradio-container {height: auto !important; min-height: 100vh !important; overflow: auto !important;}
    .gradio-container {padding: 10px !important;}
    .kpis {grid-template-columns: repeat(2, minmax(0, 1fr));}
    #topbar {flex-wrap: wrap !important; padding: 12px;}
}
"""

IMG_H = "42vh"
LOGO = ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M8 3 4 21M16 3l4 18"/><path d="M12 4v3M12 11v3M12 18v3"/></svg>')


def kpi_html(count="-", top="-", mean_conf="-", ms="-", state=("Waiting for image", "#64748b")):
    label, color = state
    cells = [(f'<span class="dot"></span>{label}', "Status", color), (count, "Detections", "#3b82f6"),
             (top, "Main damage type", color_of(top) if top in CLASS_NAMES else "#3b82f6"),
             (mean_conf, "Mean confidence", "#06d6a0"), (ms, "Inference time", "#ffd166")]
    inner = "".join(f'<div class="kpi" style="--accent:{c}"><div class="v">{v}</div><div class="l">{l}</div></div>'
                    for v, l, c in cells)
    return f'<div class="kpis">{inner}</div>'


def status_of(n):
    if n == 0:
        return ("No damage found", "#22c55e")
    return (f"Damage found" if n < 5 else "Heavy damage", "#f59e0b" if n < 5 else "#ef4444")


def detect(image, conf, iou, classes):
    """image: RGB numpy array. Returns annotated image, KPI cards, counts chart data, table, CSV path."""
    empty_counts = pd.DataFrame(columns=COUNT_COLS)
    empty_tbl = pd.DataFrame(columns=COLUMNS)
    if image is None:
        gr.Warning("Upload or capture an image first.")
        return None, kpi_html(), empty_counts, empty_tbl, None

    if image.ndim == 3 and image.shape[-1] == 4:      # drop alpha (pasted PNG)
        image = image[..., :3]
    bgr = image[..., ::-1]                            # Ultralytics expects BGR arrays

    class_ids = [NAME_TO_ID[c] for c in classes] if classes else None
    r = model.predict(bgr, conf=conf, iou=iou, imgsz=640, classes=class_ids, verbose=False)[0]

    rows = []
    for b in r.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        rows.append([model.names[int(b.cls)], round(float(b.conf), 3), x1, y1, x2, y2])
    annotated = draw_boxes(image, [tuple(x) for x in rows])
    table = pd.DataFrame(rows, columns=COLUMNS).sort_values("Confidence", ascending=False)
    ms = f"{r.speed['inference']:.0f} ms"

    if not rows:
        gr.Info(f"No damage found at confidence >= {conf:.2f}. Try lowering the threshold.")
        return annotated, kpi_html("0", "none", "-", ms, status_of(0)), empty_counts, empty_tbl, None

    counts = Counter(row[0] for row in rows)
    counts_df = pd.DataFrame(counts.most_common(), columns=COUNT_COLS)
    mean_conf = f"{table['Confidence'].mean():.2f}"
    csv_path = os.path.join(tempfile.mkdtemp(), "detections.csv")
    table.to_csv(csv_path, index=False)
    return annotated, kpi_html(str(len(rows)), counts.most_common(1)[0][0], mean_conf, ms, status_of(len(rows))), \
        counts_df, table, csv_path


def clear_all():
    return None, None, kpi_html(), pd.DataFrame(columns=COUNT_COLS), pd.DataFrame(columns=COLUMNS), None


def show_panel(which):
    """Open one overlay panel (or close all when which is None)."""
    return [gr.update(visible=(which == n)) for n in ("settings", "results", "examples", "about")]


with gr.Blocks(title="Road Damage Detection") as demo:
    # ------------------------------------------------------------- top bar
    with gr.Row(elem_id="topbar"):
        gr.HTML('<div id="title"><div class="logo">' + LOGO + '</div><div><h1>Road Damage Detection</h1>'
                '<div class="badges"><span class="badge">YOLOv10s</span><span class="badge">7 damage types</span>'
                '<span class="badge">mAP50 0.62</span><span class="badge">16.5 MB</span>'
                '<span class="badge">~100 FPS on GPU</span></div></div></div>')
        b_set = gr.Button("⚙", scale=0, variant="secondary")
        b_res = gr.Button("📊", scale=0, variant="secondary")
        b_ex = gr.Button("🖼", scale=0, variant="secondary")
        b_abt = gr.Button("ℹ", scale=0, variant="secondary")
    out_kpi = gr.HTML(kpi_html())

    # ----------------------------------------------------------- main view
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=300, elem_classes="workspace-panel"):
            inp = gr.Image(type="numpy", label="Road image", height=IMG_H, show_label=False,
                           sources=["upload", "webcam", "clipboard"])
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
            with gr.Row():
                btn = gr.Button("Detect", variant="primary", scale=3)
                clr = gr.Button("Clear", scale=1)
        with gr.Column(scale=5, min_width=300, elem_classes="workspace-panel"):
            with gr.Tabs():
                with gr.Tab("Result"):
                    out_img = gr.Image(height=IMG_H, interactive=False, show_label=False)
                    gr.HTML(legend_html())
                with gr.Tab("Detections"):
                    out_tbl = gr.Dataframe(headers=COLUMNS, interactive=False, max_height=260)
                    out_file = gr.File(label="Download CSV", height=60)
                with gr.Tab("Chart"):
                    out_plot = gr.BarPlot(pd.DataFrame(columns=COUNT_COLS), x="Damage type", y="Count",
                                          title="Detections per damage type", height=300)

    EX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
    ex_files = sorted(os.path.join(EX_DIR, f) for f in (os.listdir(EX_DIR) if os.path.isdir(EX_DIR) else [])
                      if f.lower().endswith((".jpg", ".jpeg", ".png")))[:8]
    with gr.Column(visible=False, elem_classes="overlay") as p_ex:
        gr.Markdown("### 🖼 Example images")
        if ex_files:
            gr.Examples(ex_files, inputs=inp, label="Click an image to load it, then press Detect", examples_per_page=8)
        else:
            gr.Markdown("<span class='note'>Put a few road photos (.jpg / .png) in a folder named "
                        "<b>examples</b> next to app.py and restart the app.</span>")
        c_ex = gr.Button("Close")

    # ------------------------------------------------------ overlay: settings
    with gr.Column(visible=False, elem_classes="overlay") as p_set:
        gr.Markdown("### ⚙ Detection settings")
        iou = gr.Slider(0.1, 0.9, value=0.7, step=0.05, label="IoU threshold",
                        info="Overlap limit used to merge duplicate boxes.")
        classes = gr.CheckboxGroup(CLASS_NAMES, value=CLASS_NAMES, label="Damage types to show")
        gr.Markdown("<span class='note'>Boxes are in pixels of the original image: (x1, y1) top-left, "
                    "(x2, y2) bottom-right.</span>")
        c_set = gr.Button("Close")

    # ------------------------------------------------------ overlay: results
    with gr.Column(visible=False, elem_classes="overlay") as p_res:
        gr.Markdown("### 📊 Model comparison (test set, 681 images)")
        with gr.Tabs():
            with gr.Tab("Charts"):
                with gr.Row():
                    gr.BarPlot(MAP_CHART, x="Run", y="mAP50-95", title="Test mAP50-95 by run",
                               x_title="", x_label_angle=-30, y_title="mAP50-95", y_lim=[0, 0.45], sort="-y", height=250)
                    gr.BarPlot(PER_CLASS, x="Class", y="AP50", title="Per-class AP50 (final model)",
                               x_title="", y_title="AP50", y_lim=[0, 1], sort="-y", height=250)
            with gr.Tab("Round 1: 3 models x 2 optimizers"):
                gr.Dataframe(ROUND1, interactive=False, max_height=260)
                gr.Markdown("<span class='note'>40 epochs, image size 640, batch 16, seed 0. Mean IoU is only "
                            "over matched detections (IoU >= 0.5), so it does not separate the models. "
                            "GPU FPS: T4, batch 1, end-to-end. CPU FPS: Kaggle CPU, batch 1.</span>")
            with gr.Tab("Round 2: tuning"):
                gr.Dataframe(ROUND2, interactive=False, max_height=200)
                gr.Markdown("<span class='note'>YOLOv10s + SGD, one change at a time. Lower lr0 gave the best "
                            "test score (0.390), but the gain is within run-to-run noise (single seed). "
                            "80 epochs and image size 800 did not help.</span>")
            with gr.Tab("Takeaways"):
                gr.Markdown(
                    "- The three YOLO versions are close (0.026 mAP50-95 between best and worst). "
                    "YOLOv10s + SGD is best and also the smallest (16.5 MB).\n"
                    "- Class matters more than model: block and crack are detected well; alligator and "
                    "longitudinal cracks are weak in every model.\n"
                    "- About 100 FPS on a T4 GPU and about 7 FPS on CPU: fine for photos, not for live video."
                )
        c_res = gr.Button("Close")

    # ------------------------------------------------------- overlay: about
    with gr.Column(visible=False, elem_classes="overlay") as p_abt:
        gr.Markdown(
            "### ℹ About\n"
            "**Project:** compared YOLOv8s, YOLOv10s and YOLO11s for road damage detection, tuned the best "
            "one and deployed it here.\n\n"
            "**Data:** [road-damage](https://universe.roboflow.com/trafficsignssafeway/road-damage-l1ju7) by "
            "TrafficSignsSafeway on Roboflow (CC BY 4.0): 4,930 train / 1,146 validation / 681 test images, "
            "7 classes (`edge` is almost absent).\n\n"
            "**Final model:** YOLOv10s (NMS-free), SGD, lr0 = 0.005, 40 epochs, image size 640. "
            "Test mAP50 0.621, mAP50-95 0.390, precision 0.600, recall 0.621.\n\n"
            "**Limitations:** single seed and limited tuning (differences under 0.01 mAP50-95 are not reliable); "
            "alligator and longitudinal cracks are weak (AP50 about 0.36 and 0.27); trained on this dataset only, "
            "so other cameras, lighting or countries may lower accuracy.\n\n"
            "Models: Ultralytics YOLO. Interface: Gradio."
        )
        c_abt = gr.Button("Close")

    # -------------------------------------------------------------- events
    panels = [p_set, p_res, p_ex, p_abt]
    b_set.click(lambda: show_panel("settings"), None, panels)
    b_res.click(lambda: show_panel("results"), None, panels)
    b_ex.click(lambda: show_panel("examples"), None, panels)
    b_abt.click(lambda: show_panel("about"), None, panels)
    for c in (c_set, c_res, c_ex, c_abt):
        c.click(lambda: show_panel(None), None, panels)

    inp.upload(lambda: show_panel(None), None, panels)
    btn.click(detect, [inp, conf, iou, classes], [out_img, out_kpi, out_plot, out_tbl, out_file])
    clr.click(clear_all, None, [inp, out_img, out_kpi, out_plot, out_tbl, out_file])

if __name__ == "__main__":
    share = "--share" in sys.argv
    try:   # Gradio 6 takes theme/css in launch(); older versions take them in Blocks()
        demo.launch(share=share, theme=gr.themes.Base(primary_hue="sky", neutral_hue="slate"), css=CSS,
                    js="() => {document.body.classList.add('dark');}")
    except TypeError:
        demo.launch(share=share)