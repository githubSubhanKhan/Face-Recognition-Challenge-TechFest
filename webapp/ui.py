"""Gradio UI in the Nemo's Deep Dive theme: Face Match, Sea of Faces and the session's Dive Log.

Face Match:   Face 1 and Face 2, each captured live or uploaded -> same person or not, with the evidence
              (aligned crops, similarity vs threshold, per-backbone scores, quality flags, group-photo boxes).
Sea of Faces: one person's face + a group photo -> that person is boxed in the crowd.
Dive Log:     every result of this browser session, kept in memory only.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from .engine import MODEL_NAMES, FaceEngine, Match, NoFaceError, Sweep

HERE = Path(__file__).parent
CSS = (HERE / "style.css").read_text(encoding="utf-8")
HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&display=swap" rel="stylesheet">')

MATCH_BUTTON = "🐠 Every Fish Has a Face?"
SWEEP_BUTTON = "📡 Sonar Sweep the Crowd"
FLAG_TEXT = {"blurry": "blurry", "dark": "too dark", "overexposed": "overexposed", "low_contrast": "low contrast",
             "low_res": "small / low-res face", "profile": "turned sideways", "weak_detection": "faint detection"}
GREEN, CORAL, GREY = (61, 220, 151), (255, 93, 115), (170, 190, 205)
METER_LO, METER_HI = -0.2, 1.0
LOG_MAX = 40


def theme() -> gr.themes.Base:
    """Same deep-sea palette whether the browser is in light or dark mode."""
    vals = dict(
        body_background_fill="#041a30", body_text_color="#e8f7ff", body_text_color_subdued="#93b9cf",
        background_fill_primary="rgba(6, 34, 58, 0.66)", background_fill_secondary="rgba(4, 26, 48, 0.55)",
        block_background_fill="rgba(6, 34, 58, 0.66)", block_border_color="rgba(94, 228, 255, 0.18)",
        block_label_background_fill="rgba(4, 26, 48, 0.85)", block_label_text_color="#bfe9f7",
        block_title_text_color="#e8f7ff", border_color_primary="rgba(94, 228, 255, 0.18)",
        border_color_accent="#5ee4ff", color_accent_soft="rgba(94, 228, 255, 0.15)",
        input_background_fill="rgba(2, 16, 30, 0.6)", input_border_color="rgba(94, 228, 255, 0.2)",
        button_primary_background_fill="#ff7a3d", button_primary_background_fill_hover="#ff9f5a",
        button_primary_text_color="#ffffff", button_primary_border_color="#ff7a3d",
        button_secondary_background_fill="rgba(94, 228, 255, 0.08)",
        button_secondary_background_fill_hover="rgba(94, 228, 255, 0.16)",
        button_secondary_text_color="#e8f7ff", button_secondary_border_color="rgba(94, 228, 255, 0.3)",
        panel_background_fill="rgba(6, 34, 58, 0.5)", link_text_color="#5ee4ff", loader_color="#5ee4ff",
        table_even_background_fill="rgba(255,255,255,0.03)", table_odd_background_fill="rgba(255,255,255,0.06)",
        table_border_color="rgba(94, 228, 255, 0.18)",
    )
    t = gr.themes.Base(primary_hue="orange", secondary_hue="cyan", neutral_hue="slate", radius_size="lg",
                       font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "system-ui", "sans-serif"])
    return t.set(**vals, **{k + "_dark": v for k, v in vals.items()}, color_accent="#5ee4ff",
                 block_radius="20px", button_large_radius="999px")


# ---------------------------------------------------------------------------- drawing helpers
def b64(rgb: np.ndarray, max_side: int | None = None) -> str:
    if max_side and max(rgb.shape[:2]) > max_side:
        s = max_side / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def crop_rgb(face) -> np.ndarray:
    return cv2.cvtColor(face.crop, cv2.COLOR_BGR2RGB)


def draw_faces(rgb: np.ndarray, faces: list, colors: dict, labels: dict) -> np.ndarray:
    """Numbered box on every face; faces in `colors` are highlighted, the rest are grey."""
    out = rgb.copy()
    lw = max(2, round(max(out.shape[:2]) / 320))
    fs = max(0.6, max(out.shape[:2]) / 750)
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = (int(round(v)) for v in f.bbox[:4])
        c = colors.get(i, GREY)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, lw * (2 if i in colors else 1))
        text = labels.get(i, f"#{i + 1}")
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, lw - 1))
        ty = max(th + base + 4, y1)
        cv2.rectangle(out, (x1, ty - th - base - 6), (x1 + tw + 10, ty), c, -1)
        cv2.putText(out, text, (x1 + 5, ty - base - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (4, 26, 48), max(1, lw - 1),
                    cv2.LINE_AA)
    return out


def meter_pos(x: float) -> float:
    return 100 * min(max((x - METER_LO) / (METER_HI - METER_LO), 0.0), 1.0)


def meter(score: float, thr: float, good: bool) -> str:
    return (f'<div class="dd-meter"><div class="dd-meter-track">'
            f'<div class="dd-meter-fill {"dd-good" if good else "dd-bad"}" style="width:{meter_pos(score):.1f}%"></div>'
            f'<div class="dd-meter-thr" style="left:{meter_pos(thr):.1f}%"><span>threshold {thr:.2f}</span></div></div>'
            f'<div class="dd-meter-scale"><span>different people</span><span>same person</span></div></div>')


def quality_html(name: str, face) -> str:
    chips = "".join(f'<span class="dd-chip">{FLAG_TEXT.get(f, f.replace("_", " "))}</span>' for f in face.flags) \
        or '<span class="dd-chip dd-ok">clear face</span>'
    return f'<div class="dd-row"><span>{name}</span><span>{chips}</span><span class="dd-num">{face.quality:.0%}</span></div>'


# ---------------------------------------------------------------------------- cards
def card_idle(text: str) -> str:
    return (f'<div class="dd-card dd-idle"><div class="dd-sonar"><span></span><span></span><span></span>'
            f'<i class="dd-sweep"></i><b>🐠</b></div><p class="dd-muted">{text}</p></div>')


def card_scanning(title: str, text: str) -> str:
    return (f'<div class="dd-card dd-scan"><div class="dd-sonar"><span></span><span></span><span></span>'
            f'<i class="dd-sweep"></i><b>🐟</b></div><p class="dd-title dd-dots">{title}</p>'
            f'<p class="dd-muted">{text}</p></div>')


def card_warn(title: str, text: str) -> str:
    return f'<div class="dd-card dd-warn"><div class="dd-verdict">🫧 {title}</div><p class="dd-quip">{text}</p></div>'


def calib_note(engine: FaceEngine) -> str:
    if engine.calibrated:
        return '<p class="dd-note">Threshold calibrated on our validation divers (results/calibration.json).</p>'
    return ('<p class="dd-note">Using the default threshold of 0.30: copy <code>results/calibration.json</code> from '
            'the Colab run into <code>results/</code> to use the calibrated one.</p>')


def card_match(engine: FaceEngine, m: Match, rgb_a: np.ndarray, rgb_b: np.ndarray) -> str:
    a, b = m.faces_a[m.ia], m.faces_b[m.ib]
    margin = m.score - m.threshold
    cls = "dd-same" if m.same else "dd-diff"
    verdict = "✅ SAME PERSON" if m.same else "❌ DIFFERENT PEOPLE"
    quip = ("Same fish! This face was never forgotten." if m.same
            else "Two different fish in the sea.")
    call = ('<span class="dd-pill">clear call</span>' if abs(margin) >= 0.10
            else '<span class="dd-pill dd-close">close call: near the threshold</span>')

    per = "".join(
        f'<div class="dd-row"><span>{MODEL_NAMES.get(k, k)}</span>'
        f'<div class="dd-bar"><div style="width:{meter_pos(v):.1f}%"></div></div><span class="dd-num">{v:.3f}</span></div>'
        for k, v in m.per_model.items())

    groups = ""
    for name, rgb, faces, idx in (("Face 1", rgb_a, m.faces_a, m.ia), ("Face 2", rgb_b, m.faces_b, m.ib)):
        if len(faces) > 1:
            img = draw_faces(rgb, faces, {idx: GREEN if m.same else CORAL}, {})
            groups += (f'<figure><img src="{b64(img, 640)}"><figcaption>{name}: group photo, {len(faces)} faces, '
                       f'#{idx + 1} matched best</figcaption></figure>')
    if groups:
        groups = f'<h4>Group photos</h4><div class="dd-group">{groups}</div>'

    return f"""<div class="dd-card {cls}">
  <div class="dd-verdict">{verdict}</div>
  <p class="dd-quip">{quip}</p>{call}
  <div class="dd-pair">
    <figure><img src="{b64(crop_rgb(a))}"><figcaption>Face 1 (aligned)</figcaption></figure>
    <div class="dd-link">{"=" if m.same else "≠"}</div>
    <figure><img src="{b64(crop_rgb(b))}"><figcaption>Face 2 (aligned)</figcaption></figure>
  </div>
  {meter(m.score, m.threshold, m.same)}
  <div class="dd-stats">
    <div class="dd-stat"><b>{m.score:.3f}</b><span>similarity</span></div>
    <div class="dd-stat"><b>{m.threshold:.3f}</b><span>threshold</span></div>
    <div class="dd-stat"><b>{margin:+.3f}</b><span>margin</span></div>
  </div>
  <div class="dd-details">
    <h4>Each backbone's cosine similarity</h4>{per}
    <h4>Image quality</h4>{quality_html("Face 1", a)}{quality_html("Face 2", b)}
    {groups}
  </div>
  {calib_note(engine)}
</div>"""


def card_sweep(engine: FaceEngine, sw: Sweep) -> str:
    hits = np.flatnonzero(sw.matches)
    best = int(np.argmax(sw.scores))
    n = len(sw.crowd)
    if len(hits):
        cls, verdict = "dd-same", "✅ FOUND IN THE CROWD"
        quip = (f"Spotted! Face #{best + 1} is the one you're looking for." if len(hits) == 1
                else f"{len(hits)} faces pass the threshold; face #{best + 1} is the closest match.")
    else:
        cls, verdict = "dd-diff", "❌ NOT IN THIS SEA"
        quip = f"The closest face, #{best + 1}, scored {sw.scores[best]:.3f}: below the threshold."
    order = np.argsort(-sw.scores)[:12]
    rows = "".join(
        f'<div class="dd-item"><img src="{b64(crop_rgb(sw.crowd[i]))}"><b>#{i + 1}</b>'
        f'<div class="dd-bar"><div style="width:{meter_pos(sw.scores[i]):.1f}%"></div></div>'
        f'<span class="dd-num">{sw.scores[i]:.3f}</span>'
        f'{"<span class=dd-tag-yes>match</span>" if sw.matches[i] else "<span class=dd-tag-no>other fish</span>"}</div>'
        for i in order)
    more = f'<p class="dd-muted">…and {n - len(order)} more faces.</p>' if n > len(order) else ""
    return f"""<div class="dd-card {cls}">
  <div class="dd-verdict">{verdict}</div>
  <p class="dd-quip">{quip}</p>
  <p class="dd-muted">{n} face{"s" if n != 1 else ""} swept</p>
  <div class="dd-pair"><figure><img src="{b64(crop_rgb(sw.query))}"><figcaption>The face to find</figcaption></figure></div>
  {meter(float(sw.scores[best]), float(sw.thresholds[best]), bool(len(hits)))}
  <div class="dd-list">{rows}</div>{more}
  {calib_note(engine)}
</div>"""


# ---------------------------------------------------------------------------- dive log
def log_thumb(left: np.ndarray, right: np.ndarray, ok: bool) -> np.ndarray:
    """Two faces side by side on a green (match) or coral (no match) strip."""
    size, pad = 112, 6
    tiles = [cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA) for x in (left, right)]
    canvas = np.zeros((size + 2 * pad, 2 * size + 3 * pad, 3), np.uint8)
    canvas[:] = GREEN if ok else CORAL
    canvas[pad:pad + size, pad:pad + size] = tiles[0]
    canvas[pad:pad + size, 2 * pad + size:2 * pad + 2 * size] = tiles[1]
    return canvas


def log_view(log: list):
    same = sum(e["kind"] == "match" and e["ok"] for e in log)
    diff = sum(e["kind"] == "match" and not e["ok"] for e in log)
    found = sum(e["kind"] == "sweep" and e["ok"] for e in log)
    sweeps = sum(e["kind"] == "sweep" for e in log)
    stats = (f'<div class="dd-logstats"><div><b>{len(log)}</b><span>dives</span></div>'
             f'<div><b>{same}</b><span>same person</span></div><div><b>{diff}</b><span>different</span></div>'
             f'<div><b>{found}/{sweeps}</b><span>found in crowd</span></div></div>')
    if not log:
        stats += '<p class="dd-intro">No dives yet. Results from Face Match and Sea of Faces appear here.</p>'
    return stats, [(e["img"], e["caption"]) for e in log]


def log_add(log: list, img: np.ndarray, caption: str, kind: str, ok: bool) -> list:
    return ([{"img": img, "caption": f"{time.strftime('%H:%M:%S')} · {caption}", "kind": kind, "ok": ok}] + log)[:LOG_MAX]


# ---------------------------------------------------------------------------- app
BUBBLES = '<div class="dd-ocean">' + "".join(
    f'<i style="left:{x}%;width:{s}px;height:{s}px;animation-duration:{d}s;animation-delay:-{dl}s"></i>'
    for x, s, d, dl in [(4, 10, 16, 2), (11, 18, 22, 9), (19, 8, 13, 5), (27, 14, 19, 12), (36, 6, 11, 3),
                        (44, 22, 26, 15), (53, 9, 15, 7), (61, 16, 21, 1), (69, 7, 12, 10), (77, 20, 24, 4),
                        (85, 11, 17, 13), (92, 15, 20, 6), (97, 8, 14, 11)]) + "</div>"

HERO = """<div class="dd-hero">
  <div class="dd-kicker">AI Odyssey · Finding the Unknown · Tech Fest '26</div>
  <h1>NEMO'S <span>DEEP</span> DIVE</h1>
  <div class="dd-sub">Every Fish Has a Face</div>
  <p class="dd-tag">Go deeper. Never forget a face.</p>
  <svg class="dd-wave" viewBox="0 0 1200 26" preserveAspectRatio="none" aria-hidden="true">
    <path d="M0 13 C150 0 250 26 400 13 S650 0 800 13 S1050 26 1200 13" fill="none" stroke="#5ee4ff" stroke-width="2"/>
  </svg>
</div>"""

FOOTER = """<div class="dd-footer">
  <b>PakFace</b> · SCRFD detection → 5-point alignment → AdaFace IR-101 + ArcFace R100 ensemble · cosine similarity<br>
  Photos are processed in memory for this session only; the server's upload cache is cleared every 10 minutes.<br>
  See you beneath the surface. 🫧
</div>"""


def build(engine: FaceEngine) -> gr.Blocks:
    def image_box(label: str, first: str = "webcam") -> gr.Image:
        sources = ["webcam", "upload"] if first == "webcam" else ["upload", "webcam"]
        return gr.Image(label=label, sources=sources, type="numpy", image_mode="RGB", height=380,
                        webcam_options=gr.WebcamOptions(mirror=True), elem_classes="dd-cam")

    def run_match(rgb_a, rgb_b, log):
        if rgb_a is None or rgb_b is None:
            missing = " and ".join(n for n, x in (("Face 1", rgb_a), ("Face 2", rgb_b)) if x is None)
            return card_warn("The tank is empty", f"{missing}: capture a live picture or upload a photo first."), log, *log_view(log)
        try:
            m = engine.compare(rgb_a, rgb_b)
        except NoFaceError as e:
            return card_warn(f"No face surfaced in {e}",
                             "Face the camera, add some light or move a little closer, then try again."), log, *log_view(log)
        thumb = log_thumb(crop_rgb(m.faces_a[m.ia]), crop_rgb(m.faces_b[m.ib]), m.same)
        log = log_add(log, thumb, f"{'✅ same' if m.same else '❌ different'} · {m.score:.3f}", "match", m.same)
        return card_match(engine, m, rgb_a, rgb_b), log, *log_view(log)

    def run_sweep(query, crowd, log):
        if query is None or crowd is None:
            missing = " and ".join(n for n, x in (("the face to find", query), ("the group photo", crowd)) if x is None)
            return None, card_warn("The sea is empty", f"Add {missing} first."), log, *log_view(log)
        try:
            sw = engine.sweep(query, crowd)
        except NoFaceError as e:
            return None, card_warn(f"No face surfaced in {e}",
                                   "Try a clearer, brighter photo where faces are not too small."), log, *log_view(log)
        colors = {int(i): GREEN for i in np.flatnonzero(sw.matches)}
        labels = {i: f"#{i + 1} {s:.2f}" for i, s in enumerate(sw.scores)}
        annotated = draw_faces(crowd, sw.crowd, colors, labels)
        found = bool(sw.matches.any())
        best = int(np.argmax(sw.scores))
        thumb = log_thumb(crop_rgb(sw.query), crop_rgb(sw.crowd[best]), found)
        caption = f"{'✅ found' if found else '❌ not found'} in crowd of {len(sw.crowd)} · {sw.scores[best]:.3f}"
        log = log_add(log, thumb, caption, "sweep", found)
        return annotated, card_sweep(engine, sw), log, *log_view(log)

    with gr.Blocks(title="Nemo's Deep Dive · Face Match", delete_cache=(600, 600)) as demo:
        gr.HTML(BUBBLES + HERO, padding=False)
        log = gr.State([])
        with gr.Tabs():
            with gr.Tab("🐠 Face Match"):
                gr.HTML('<p class="dd-intro">Capture a <b>live picture</b> or <b>upload a photo</b> for each side, '
                        'then ask the deep: is it the same face?</p>')
                with gr.Row(equal_height=True, elem_classes="dd-split"):
                    img_a = image_box("Face 1")
                    img_b = image_box("Face 2")
                with gr.Row(elem_classes=["dd-actions", "dd-split"]):
                    dive = gr.Button(MATCH_BUTTON, variant="primary", size="lg", elem_id="dive-btn", scale=4)
                    reset = gr.Button("↺ New dive", variant="secondary", size="lg", scale=1)
                match_out = gr.HTML(card_idle("Two faces in, one answer out. The result surfaces here."))

            with gr.Tab("🌊 Sea of Faces"):
                gr.HTML('<p class="dd-intro"><b>A sea of faces. One to find.</b> Give one person\'s face and a '
                        'group or classroom photo: the sonar boxes that person in the crowd.</p>')
                with gr.Row(equal_height=True, elem_classes="dd-split"):
                    query = image_box("The face to find")
                    crowd = image_box("The sea of faces (group photo)", first="upload")
                with gr.Row(elem_classes=["dd-actions", "dd-split"]):
                    sweep = gr.Button(SWEEP_BUTTON, variant="primary", size="lg", elem_id="sweep-btn", scale=4)
                    reset_sea = gr.Button("↺ New sweep", variant="secondary", size="lg", scale=1)
                sweep_img = gr.Image(label="Sonar result: green = match, grey = other faces", interactive=False,
                                     visible=True, height=460, elem_classes="dd-cam")
                sweep_out = gr.HTML(card_idle("Every fish has a face. The sweep result surfaces here."))

            with gr.Tab("📜 Dive Log"):
                stats0, _ = log_view([])
                log_stats = gr.HTML(stats0)
                gallery = gr.Gallery(show_label=False, columns=4, height="auto", object_fit="contain",
                                     allow_preview=True)
                clear_log = gr.Button("Clear the log", variant="secondary", size="sm")
        gr.HTML(FOOTER)

        dive.click(lambda: card_scanning("Sending sonar pings", "Detecting, aligning and comparing both faces"),
                   outputs=match_out, show_progress="hidden", queue=False
                   ).then(run_match, [img_a, img_b, log], [match_out, log, log_stats, gallery], show_progress="hidden")
        reset.click(lambda: (None, None, card_idle("Two faces in, one answer out. The result surfaces here.")),
                    outputs=[img_a, img_b, match_out], queue=False)
        sweep.click(lambda: card_scanning("Sweeping the sea of faces", "Every face in the photo is being compared"),
                    outputs=sweep_out, show_progress="hidden", queue=False
                    ).then(run_sweep, [query, crowd, log], [sweep_img, sweep_out, log, log_stats, gallery],
                           show_progress="hidden")
        reset_sea.click(lambda: (None, None, None, card_idle("Every fish has a face. The sweep result surfaces here.")),
                        outputs=[query, crowd, sweep_img, sweep_out], queue=False)
        clear_log.click(lambda: ([], *log_view([])), outputs=[log, log_stats, gallery], queue=False)
    return demo
