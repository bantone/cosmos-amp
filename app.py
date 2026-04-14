"""
Cosmos Reason2-8B — Streamlit Visual Q&A App
nvidia/Cosmos-Reason2-8B loaded locally via HuggingFace Transformers.
"""

import pathlib
import re
import tempfile
import threading
import time

import cv2
import torch
import transformers
import streamlit as st
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME    = "nvidia/Cosmos-Reason2-8B"
MAX_NEW_TOKENS = 4096
VIDEO_FPS      = 4

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
ALL_EXTENSIONS   = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

SYSTEM_MESSAGE = {
    "role": "system",
    "content": [{"type": "text", "text": "You are a helpful assistant."}],
}

LOAD_STEPS = ["resolving", "config", "weights", "device", "processor"]
STEP_LABELS = {
    "resolving":  "Resolving model files from HuggingFace Hub",
    "config":     "Reading model configuration",
    "weights":    "Loading model weights into memory",
    "device":     "Initialising model on device",
    "processor":  "Loading tokenizer & processor",
}

# ── Thread-safe module-level state ─────────────────────────────────────────────
# st.session_state cannot be written from background threads — Streamlit raises
# "missing ScriptRunContext" and drops the write silently.  We keep a plain dict
# at module level instead; the background thread writes here and the main thread
# reads it on every rerun.
_LOAD_LOCK  = threading.Lock()
_LOAD: dict = {
    "stage":        None,   # None | step-key | "ready" | "error"
    "steps_done":   [],     # list of (step_key, elapsed_str)
    "error":        None,
    "start_time":   None,
    "model":        None,
    "processor":    None,
    "device_info":  "",
    "thread_alive": False,
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cosmos Reason2-8B",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d0f14;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #111318;
    border-right: 1px solid #1e2230;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f1a2e 0%, #0d1117 60%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 26px 32px 22px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 18px;
}
.hero-icon { font-size: 2.8rem; line-height: 1; }
.hero-title {
    font-size: 1.75rem;
    font-weight: 700;
    background: linear-gradient(90deg, #76b9f7, #a78bfa, #60e0cf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 0.78rem;
    color: #4a5568;
    margin-top: 4px;
    font-family: monospace;
    letter-spacing: 0.04em;
}

/* Loading card */
.load-card {
    background: linear-gradient(160deg, #0f1a2e 0%, #111318 100%);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 36px 44px;
    max-width: 660px;
    margin: 32px auto;
}
.load-title {
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #76b9f7, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
.load-subtitle { font-size: 0.78rem; color: #4a5568; font-family: monospace; margin-bottom: 24px; }
.load-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 9px 0;
    border-bottom: 1px solid #1a1f2e;
    font-size: 0.86rem;
}
.load-step:last-child { border-bottom: none; }
.step-icon {
    width: 24px; height: 24px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; flex-shrink: 0; font-weight: 700;
}
.step-done   { background:rgba(34,197,94,.14);  border:1px solid rgba(34,197,94,.4);  color:#4ade80; }
.step-active { background:rgba(74,144,217,.14); border:1px solid rgba(74,144,217,.4); color:#76b9f7; }
.step-wait   { background:rgba(100,116,139,.1); border:1px solid rgba(100,116,139,.2);color:#334155; }
.label-done   { color:#64748b; }
.label-active { color:#e2e8f0; font-weight:500; }
.label-wait   { color:#334155; }
.step-time { margin-left:auto; font-size:0.7rem; font-family:monospace; color:#334155; flex-shrink:0; }
.load-meta {
    margin-top: 18px; padding-top: 14px;
    border-top: 1px solid #1a1f2e;
    display: flex; gap: 20px; flex-wrap: wrap;
}
.load-meta-item { font-size: 0.74rem; color: #4a5568; font-family: monospace; }
.load-meta-item span { color: #76b9f7; }
.load-error {
    background:rgba(239,68,68,.08); border:1px solid rgba(239,68,68,.25);
    border-radius:10px; padding:12px 16px; color:#f87171;
    font-size:0.8rem; margin-top:14px; font-family:monospace;
    white-space:pre-wrap; word-break:break-word;
}

/* Section labels */
.section-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.13em;
    text-transform: uppercase; color: #4a90d9; margin-bottom: 8px; margin-top: 2px;
}

/* Status pill */
.status-pill {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 6px 14px; border-radius: 20px;
    font-size: 0.82rem; font-weight: 500;
    width: 100%; justify-content: center; box-sizing: border-box;
}
.status-ready   { background:rgba(34,197,94,.12);  border:1px solid rgba(34,197,94,.35);  color:#4ade80; }
.status-loading { background:rgba(251,191,36,.10); border:1px solid rgba(251,191,36,.30); color:#fbbf24; }
.status-error   { background:rgba(239,68,68,.10);  border:1px solid rgba(239,68,68,.30);  color:#f87171; }

/* File count */
.file-count {
    background:rgba(74,144,217,.1); border:1px solid rgba(74,144,217,.22);
    border-radius:8px; padding:5px 12px; font-size:0.76rem; color:#76b9f7; text-align:center;
}

/* Media card */
.media-card {
    background:#111318; border:1px solid #1e2230;
    border-radius:14px; padding:16px 16px 12px;
}
.media-card-title {
    font-size:0.68rem; font-weight:600; letter-spacing:0.1em;
    text-transform:uppercase; color:#4a90d9; margin-bottom:12px;
}
.media-filename {
    font-size:0.75rem; color:#64748b; font-family:monospace;
    margin-top:8px; text-align:center; word-break:break-all; padding:0 4px;
}
.media-badge {
    display:inline-block; padding:2px 10px; border-radius:10px;
    font-size:0.68rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;
}
.media-badge-video { background:rgba(167,139,250,.15); border:1px solid rgba(167,139,250,.35); color:#a78bfa; }
.media-badge-image { background:rgba(96,224,207,.10);  border:1px solid rgba(96,224,207,.28);  color:#60e0cf; }
.nav-counter { font-size:0.8rem; color:#64748b; font-family:monospace; text-align:center; padding:2px 0; }
.media-placeholder {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    min-height:220px; color:#1e2230; font-size:0.88rem; gap:10px;
}

/* Chat card */
.chat-card { background:#111318; border:1px solid #1e2230; border-radius:14px; padding:16px 20px; }
.chat-card-title {
    font-size:0.68rem; font-weight:600; letter-spacing:0.1em;
    text-transform:uppercase; color:#4a90d9; margin-bottom:14px;
}
.chat-empty {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    min-height:160px; color:#334155; font-size:0.88rem; gap:9px; text-align:center;
}

[data-testid="stExpander"] {
    background:rgba(167,139,250,.05) !important;
    border:1px solid rgba(167,139,250,.2) !important;
    border-radius:10px !important;
}
[data-testid="stExpander"] summary { color:#a78bfa !important; font-size:0.8rem !important; }
[data-testid="stTextInput"] input {
    background:#0d0f14 !important; border:1px solid #1e2230 !important;
    border-radius:8px !important; color:#e2e8f0 !important; font-size:0.85rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color:#4a90d9 !important; box-shadow:0 0 0 2px rgba(74,144,217,.2) !important;
}
[data-testid="stFileUploader"] {
    background:rgba(74,144,217,.04) !important;
    border:1px dashed #1e3a5f !important; border-radius:10px !important;
}
[data-testid="stChatMessage"] { background:transparent !important; }
hr { border-color:#1e2230 !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0d0f14; }
::-webkit-scrollbar-thumb { background:#1e2230; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in {
    "media_files":  [],
    "media_index":  0,
    "history":      [],
    "generating":   False,
    "video_fps":    VIDEO_FPS,
    "max_new_tokens": MAX_NEW_TOKENS,
    "upload_dir":   None,
    "last_folder":  "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Model loading (background thread → module dict) ────────────────────────────

def _advance(step_key: str) -> None:
    """Record the current step as done and move to the next one."""
    with _LOAD_LOCK:
        prev = _LOAD["stage"]
        if prev and prev not in ("error", "ready"):
            elapsed = f"{time.time() - _LOAD['start_time']:.1f}s"
            _LOAD["steps_done"].append((prev, elapsed))
        _LOAD["stage"] = step_key


def _load_model_thread() -> None:
    try:
        with _LOAD_LOCK:
            _LOAD["start_time"] = time.time()
            _LOAD["steps_done"]  = []
            _LOAD["error"]       = None

        # 1 · Resolve
        _advance("resolving")

        # 2 · Config
        _advance("config")
        transformers.AutoConfig.from_pretrained(MODEL_NAME)

        # 3 · Weights
        _advance("weights")
        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )

        # 4 · Device info
        _advance("device")
        device    = str(next(model.parameters()).device)
        info_parts = [f"device: {device}"]
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            info_parts += [f"GPU: {gpu_name}", f"VRAM: {vram_gb:.1f} GB"]
        with _LOAD_LOCK:
            _LOAD["device_info"] = "  ·  ".join(info_parts)

        # 5 · Processor
        _advance("processor")
        processor = transformers.AutoProcessor.from_pretrained(MODEL_NAME)

        # Done
        with _LOAD_LOCK:
            _LOAD["stage"]     = "ready"
            _LOAD["model"]     = model
            _LOAD["processor"] = processor

    except Exception as exc:
        with _LOAD_LOCK:
            _LOAD["stage"] = "error"
            _LOAD["error"] = str(exc)
    finally:
        with _LOAD_LOCK:
            _LOAD["thread_alive"] = False


def ensure_loading() -> None:
    with _LOAD_LOCK:
        if _LOAD["stage"] is None and not _LOAD["thread_alive"]:
            _LOAD["thread_alive"] = True
            threading.Thread(target=_load_model_thread, daemon=True).start()


# ── Inference helpers ──────────────────────────────────────────────────────────

def detect_media_type(path: str) -> str:
    ext = pathlib.Path(path).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    raise ValueError(f"Unsupported extension: {ext}")


def list_media_files(folder: str) -> list[str]:
    p = pathlib.Path(folder)
    if not p.is_dir():
        return []
    return sorted(
        [str(f) for f in p.iterdir() if f.suffix.lower() in ALL_EXTENSIONS],
        key=lambda s: pathlib.Path(s).name.lower(),
    )


def decode_video(path: str, fps: int = VIDEO_FPS) -> list:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or fps
    interval = max(1, round(src_fps / fps))
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    if not frames:
        raise ValueError(f"No frames extracted from: {path}")
    return frames


def extract_media(messages: list) -> tuple[list, list]:
    videos, images = [], []
    for msg in messages:
        for item in msg.get("content", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "video":
                videos.append(item["video"])
            elif item.get("type") == "image":
                images.append(item["image"])
    return videos, images


def build_user_message(text: str, media_uri: str | None = None, media_type: str | None = None) -> dict:
    content = []
    if media_uri and media_type:
        if media_type == "video":
            content.append({"type": "video", "video": media_uri, "fps": st.session_state.video_fps})
        else:
            content.append({"type": "image", "image": media_uri})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def parse_response(raw: str) -> tuple[str, str]:
    match = re.search(r"<think>(.*?)</think>(.*)", raw, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", raw.strip()


def run_inference(history: list) -> str:
    with _LOAD_LOCK:
        model     = _LOAD["model"]
        processor = _LOAD["processor"]

    messages = [SYSTEM_MESSAGE] + history
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    video_paths, image_paths = extract_media(messages)
    decoded_videos = [decode_video(p, st.session_state.video_fps) for p in video_paths] if video_paths else None
    decoded_images = [Image.open(p).convert("RGB") for p in image_paths]               if image_paths else None

    inputs = processor(text=[text], videos=decoded_videos, images=decoded_images, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=st.session_state.max_new_tokens)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids, strict=False)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def current_file() -> str | None:
    files = st.session_state.media_files
    if not files:
        return None
    return files[min(st.session_state.media_index, len(files) - 1)]


def set_media_index(idx: int) -> None:
    files = st.session_state.media_files
    if not files:
        return
    new_idx = idx % len(files)
    if new_idx != st.session_state.media_index:
        st.session_state.media_index = new_idx
        st.session_state.history = []


def set_media_files(files: list[str]) -> None:
    st.session_state.media_files = files
    st.session_state.media_index = 0
    st.session_state.history     = []


# ══════════════════════════════════════════════════════════════════════════════
# Render
# ══════════════════════════════════════════════════════════════════════════════

ensure_loading()

# Read shared state once per rerun (no lock needed for reads of primitives)
stage      = _LOAD["stage"]
steps_done = list(_LOAD["steps_done"])
model_ready = stage == "ready"

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-icon">🪐</div>
  <div>
    <p class="hero-title">Cosmos Reason2-8B</p>
    <p class="hero-sub">{MODEL_NAME} &nbsp;·&nbsp; Visual Question Answering</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Loading screen ─────────────────────────────────────────────────────────────
if not model_ready:
    done_map  = {s: t for s, t in steps_done}
    done_keys = set(done_map)

    rows = ""
    for key in LOAD_STEPS:
        label = STEP_LABELS[key]
        if key in done_keys:
            rows += (
                f'<div class="load-step">'
                f'<div class="step-icon step-done">✓</div>'
                f'<span class="label-done">{label}</span>'
                f'<span class="step-time">{done_map[key]}</span>'
                f'</div>'
            )
        elif key == stage:
            rows += (
                f'<div class="load-step">'
                f'<div class="step-icon step-active">◌</div>'
                f'<span class="label-active">{label}</span>'
                f'</div>'
            )
        else:
            rows += (
                f'<div class="load-step">'
                f'<div class="step-icon step-wait">·</div>'
                f'<span class="label-wait">{label}</span>'
                f'</div>'
            )

    elapsed = ""
    if _LOAD["start_time"]:
        elapsed = f"{time.time() - _LOAD['start_time']:.0f}s elapsed"

    meta = ""
    if _LOAD["device_info"]:
        for part in _LOAD["device_info"].split("  ·  "):
            k, _, v = part.partition(": ")
            meta += f'<div class="load-meta-item">{k}: <span>{v}</span></div>'
        meta = f'<div class="load-meta">{meta}</div>'

    error_block = ""
    if stage == "error" and _LOAD["error"]:
        error_block = f'<div class="load-error">✕  {_LOAD["error"]}</div>'

    st.markdown(
        f'<div class="load-card">'
        f'<p class="load-title">Loading Cosmos Reason2-8B</p>'
        f'<p class="load-subtitle">{MODEL_NAME} &nbsp;·&nbsp; {elapsed}</p>'
        f'{rows}{error_block}{meta}'
        f'</div>',
        unsafe_allow_html=True,
    )

    progress = min(len(steps_done) / len(LOAD_STEPS), 1.0)
    st.progress(progress)

    if stage == "error":
        if st.button("Retry", type="primary"):
            with _LOAD_LOCK:
                _LOAD["stage"]        = None
                _LOAD["steps_done"]   = []
                _LOAD["error"]        = None
                _LOAD["start_time"]   = None
                _LOAD["thread_alive"] = False
            ensure_loading()
            st.rerun()
    else:
        time.sleep(0.75)
        st.rerun()

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Main app — model is loaded
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">Model</p>', unsafe_allow_html=True)
    st.markdown('<div class="status-pill status-ready">● Cosmos Reason2-8B ready</div>', unsafe_allow_html=True)
    if _LOAD["device_info"]:
        st.caption(_LOAD["device_info"])

    st.divider()

    st.markdown('<p class="section-label">Media Source</p>', unsafe_allow_html=True)
    tab_folder, tab_upload = st.tabs(["📁  Folder", "⬆  Upload"])

    with tab_folder:
        folder_input = st.text_input(
            "folder_path", placeholder="/path/to/media",
            label_visibility="collapsed", key="folder_input",
        )
        if folder_input and folder_input != st.session_state.last_folder:
            found = list_media_files(folder_input)
            if found:
                set_media_files(found)
                st.session_state.last_folder = folder_input
            elif not pathlib.Path(folder_input).is_dir():
                st.warning("Folder not found.")
            else:
                st.info("No supported media files found.")

        if st.session_state.media_files and st.session_state.last_folder == folder_input:
            files = st.session_state.media_files
            imgs  = sum(1 for f in files if pathlib.Path(f).suffix.lower() in IMAGE_EXTENSIONS)
            vids  = sum(1 for f in files if pathlib.Path(f).suffix.lower() in VIDEO_EXTENSIONS)
            parts = ([f"{imgs} image{'s' if imgs!=1 else ''}"] if imgs else []) + \
                    ([f"{vids} video{'s' if vids!=1 else ''}"] if vids else [])
            st.markdown(f'<div class="file-count">📁 {" · ".join(parts)}</div>', unsafe_allow_html=True)

            names = [pathlib.Path(f).name for f in files]
            chosen = st.selectbox("Jump to", names, index=st.session_state.media_index, key="folder_sel")
            chosen_idx = names.index(chosen)
            if chosen_idx != st.session_state.media_index:
                set_media_index(chosen_idx)
                st.rerun()

    with tab_upload:
        uploaded = st.file_uploader(
            "upload", type=[e.lstrip(".") for e in ALL_EXTENSIONS],
            accept_multiple_files=True, label_visibility="collapsed",
        )
        if uploaded:
            if st.session_state.upload_dir is None:
                st.session_state.upload_dir = tempfile.mkdtemp(prefix="cosmos_uploads_")
            udir = pathlib.Path(st.session_state.upload_dir)
            saved = sorted(
                [str((udir / uf.name).write_bytes(uf.getvalue()) or (udir / uf.name)) for uf in uploaded],
                key=lambda s: pathlib.Path(s).name.lower(),
            )
            if saved != st.session_state.media_files:
                set_media_files(saved)
                st.rerun()
            imgs = sum(1 for f in saved if pathlib.Path(f).suffix.lower() in IMAGE_EXTENSIONS)
            vids = sum(1 for f in saved if pathlib.Path(f).suffix.lower() in VIDEO_EXTENSIONS)
            parts = ([f"{imgs} image{'s' if imgs!=1 else ''}"] if imgs else []) + \
                    ([f"{vids} video{'s' if vids!=1 else ''}"] if vids else [])
            st.markdown(f'<div class="file-count">⬆ {" · ".join(parts)} uploaded</div>', unsafe_allow_html=True)

    st.divider()

    if st.session_state.media_files:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.divider()

    st.markdown('<p class="section-label">Inference Settings</p>', unsafe_allow_html=True)
    st.session_state.video_fps = st.slider(
        "Video sample rate (fps)", 1, 10, st.session_state.video_fps,
        help="Frames sampled per second when processing videos.",
    )
    st.session_state.max_new_tokens = st.slider(
        "Max new tokens", 256, 8192, st.session_state.max_new_tokens, step=256,
    )


# ── Main columns ───────────────────────────────────────────────────────────────
col_media, col_chat = st.columns([5, 7], gap="large")
active_file = current_file()

# ── Media panel ───────────────────────────────────────────────────────────────
with col_media:
    st.markdown('<div class="media-card">', unsafe_allow_html=True)
    st.markdown('<p class="media-card-title">Media Preview</p>', unsafe_allow_html=True)

    if active_file:
        path       = pathlib.Path(active_file)
        media_type = detect_media_type(active_file)

        if media_type == "image":
            st.image(Image.open(path), use_container_width=True)
        else:
            st.video(active_file)

        badge_cls  = "media-badge-video" if media_type == "video" else "media-badge-image"
        badge_icon = "▶" if media_type == "video" else "◼"
        st.markdown(
            f'<div style="text-align:center;margin-top:6px;">'
            f'<span class="media-badge {badge_cls}">{badge_icon} {media_type}</span>'
            f'<p class="media-filename">{path.name}</p></div>',
            unsafe_allow_html=True,
        )

        files = st.session_state.media_files
        total = len(files)
        idx   = st.session_state.media_index

        if total > 1:
            st.markdown(f'<p class="nav-counter">{idx+1} / {total}</p>', unsafe_allow_html=True)
            c_prev, c_mid, c_next = st.columns([1, 2, 1])
            with c_prev:
                if st.button("← Prev", use_container_width=True):
                    set_media_index(idx - 1); st.rerun()
            with c_mid:
                start = max(0, min(idx - 2, total - 5))
                end   = min(total, start + 5)
                tcols = st.columns(end - start)
                for ci, fi in enumerate(range(start, end)):
                    with tcols[ci]:
                        tp        = pathlib.Path(files[fi])
                        is_active = fi == idx
                        if tp.suffix.lower() in IMAGE_EXTENSIONS:
                            thumb = Image.open(tp); thumb.thumbnail((60, 60))
                            st.image(thumb, use_container_width=True)
                        else:
                            border = "2px solid #4a90d9" if is_active else "2px solid transparent"
                            st.markdown(
                                f'<div style="background:#1e2230;border:{border};border-radius:4px;'
                                f'height:40px;display:flex;align-items:center;justify-content:center;'
                                f'font-size:1.1rem;">▶</div>',
                                unsafe_allow_html=True,
                            )
                        if st.button("·", key=f"t{fi}", use_container_width=True, help=tp.name,
                                     type="primary" if is_active else "secondary"):
                            set_media_index(fi); st.rerun()
            with c_next:
                if st.button("Next →", use_container_width=True):
                    set_media_index(idx + 1); st.rerun()
    else:
        st.markdown("""
        <div class="media-placeholder">
          <span style="font-size:3rem;opacity:0.3">🎞</span>
          <span>No media loaded</span>
          <span style="font-size:0.75rem;color:#1e2230">Use the sidebar to browse a folder or upload files</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Chat panel ────────────────────────────────────────────────────────────────
with col_chat:
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    st.markdown('<p class="chat-card-title">Conversation</p>', unsafe_allow_html=True)

    if not active_file:
        st.markdown("""
        <div class="chat-empty">
          <span style="font-size:2.4rem;opacity:0.3">📂</span>
          <span>No media selected</span>
          <span style="font-size:0.75rem">Browse a folder or upload files using the sidebar</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        if not st.session_state.history:
            mtype = detect_media_type(active_file)
            fname = pathlib.Path(active_file).name
            st.markdown(f"""
            <div class="chat-empty">
              <span style="font-size:2.4rem;opacity:0.3">💬</span>
              <span>Ask anything about <code style="color:#76b9f7">{fname}</code></span>
              <span style="font-size:0.75rem">Reasoning traces shown inline · first message attaches the {mtype}</span>
            </div>
            """, unsafe_allow_html=True)

        for turn in st.session_state.history:
            if turn["role"] == "user":
                text = next(
                    (i["text"] for i in turn.get("content", [])
                     if isinstance(i, dict) and i.get("type") == "text"), "",
                )
                with st.chat_message("user"):
                    st.write(text)
            elif turn["role"] == "assistant":
                raw = turn.get("content", "") if isinstance(turn.get("content"), str) else ""
                thinking, answer = parse_response(raw)
                with st.chat_message("assistant"):
                    if thinking:
                        with st.expander("🧠 Reasoning trace"):
                            st.markdown(thinking)
                    st.markdown(answer)

        user_input = st.chat_input(
            "Ask a question about the selected media…",
            disabled=st.session_state.generating,
        )

        if user_input:
            is_first = len(st.session_state.history) == 0
            user_msg = build_user_message(
                user_input,
                media_uri  = active_file if is_first else None,
                media_type = detect_media_type(active_file) if is_first else None,
            )
            st.session_state.history.append(user_msg)

            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Running inference…"):
                    st.session_state.generating = True
                    try:
                        raw = run_inference(st.session_state.history)
                    except Exception as exc:
                        raw = f"Error during inference: {exc}"
                    finally:
                        st.session_state.generating = False

                thinking, answer = parse_response(raw)
                if thinking:
                    with st.expander("🧠 Reasoning trace"):
                        st.markdown(thinking)
                st.markdown(answer)

            st.session_state.history.append({"role": "assistant", "content": raw})
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
