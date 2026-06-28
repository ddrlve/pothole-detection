from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.feature import local_binary_pattern

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent


# ===========================================================================
# ARTIFACT LOADING
# ===========================================================================

def find_artifact_dir() -> Path:
    candidates = [
        APP_DIR / "pothole_output",
        APP_DIR,
        APP_DIR.parent / "pothole_output",
        Path.cwd() / "pothole_output",
        Path.cwd(),
    ]
    for folder in candidates:
        if (folder / "pothole_config.json").exists() and (
            (folder / "pothole_model_fast.pkl").exists()
            or (folder / "pothole_model.pkl").exists()
            or (folder / "pothole_model_accuracy.pkl").exists()
        ):
            return folder
    raise FileNotFoundError(
        "Model artifacts not found. Ensure pothole_config.json and a .pkl model file "
        "exist inside the pothole_output folder."
    )


ARTIFACT_DIR = find_artifact_dir()
CONFIG_PATH  = ARTIFACT_DIR / "pothole_config.json"

with open(CONFIG_PATH, "r") as f:
    CFG = json.load(f)

FEATURE_NAMES     = CFG["feature_names"]
DEFAULT_WORK_SIZE = int(CFG.get("app_work_size", CFG.get("work_size", 256)))
DEFAULT_THRESHOLD = float(CFG.get("best_postprocess", {}).get("threshold", 0.60))
DEFAULT_MIN_AREA  = int(CFG.get("best_postprocess", {}).get("min_area", 250))
DEFAULT_CLOSE_K   = int(CFG.get("best_postprocess", {}).get("close_k", 5))
DEFAULT_OPEN_K    = int(CFG.get("best_postprocess", {}).get("open_k", 3))
DEFAULT_FILL      = bool(CFG.get("best_postprocess", {}).get("fill", False))
DEFAULT_MAX_AREA  = float(CFG.get("best_postprocess", {}).get("max_area_ratio", 0.22))
MODEL_NOTE        = CFG.get("model_note", "Classical ML pothole segmentation")


# ===========================================================================
# ENSEMBLE MODEL
# ===========================================================================

class SoftProbabilityEnsemble:
    def __init__(self, models, model_names=None, weights=None):
        self.models      = models
        self.model_names = model_names or [f"model_{i}" for i in range(len(models))]
        self.weights     = None if weights is None else np.asarray(weights, dtype=np.float32)
        self.classes_    = np.array([0, 1], dtype=int)

    def _positive_proba(self, model, X):
        raw = np.asarray(model.predict_proba(X), dtype=np.float32)
        if raw.ndim == 1:
            return raw
        if hasattr(model, "classes_"):
            classes = np.asarray(model.classes_).astype(int)
            if 1 in classes:
                return raw[:, int(np.where(classes == 1)[0][0])]
        return raw[:, 1] if raw.shape[1] == 2 else raw[:, -1]

    def predict_proba(self, X):
        probs, weights = [], []
        for i, model in enumerate(self.models):
            if not hasattr(model, "predict_proba"):
                continue
            probs.append(self._positive_proba(model, X))
            weights.append(1.0 if self.weights is None else float(self.weights[i]))
        if not probs:
            raise RuntimeError("No valid predict_proba model in ensemble.")
        W  = np.asarray(weights, dtype=np.float32)
        W  = W / (W.sum() + 1e-8)
        p1 = np.zeros_like(probs[0], dtype=np.float32)
        for p, w in zip(probs, W):
            p1 += w * p.astype(np.float32)
        p1 = np.clip(p1, 0, 1)
        return np.vstack([1 - p1, p1]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.uint8)


@st.cache_resource(show_spinner=False)
def load_models():
    fast_path = ARTIFACT_DIR / "pothole_model_fast.pkl"
    acc_path  = ARTIFACT_DIR / "pothole_model_accuracy.pkl"
    main_path = ARTIFACT_DIR / "pothole_model.pkl"

    fast = joblib.load(fast_path) if fast_path.exists() else (
           joblib.load(main_path) if main_path.exists() else None)
    acc  = joblib.load(acc_path)  if acc_path.exists()  else (
           joblib.load(main_path) if main_path.exists() else fast)

    if fast is None and acc is None:
        raise FileNotFoundError("No model .pkl file found in artifact directory.")
    return fast, acc


# ===========================================================================
# IMAGE / FEATURE UTILITIES
# ===========================================================================

def read_uploaded_image(uploaded_file):
    data = uploaded_file.getvalue()
    try:
        pil = Image.open(__import__("io").BytesIO(data))
        pil = ImageOps.exif_transpose(pil)
        if pil.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil, mask=pil.split()[-1])
            pil = bg
        else:
            pil = pil.convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, Exception):
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def normalize01(x):
    x  = x.astype(np.float32)
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn + 1e-8)).astype(np.float32)


def local_mean_std(gray, k=7):
    gray    = gray.astype(np.float32)
    mean    = cv2.blur(gray, (k, k))
    mean_sq = cv2.blur(gray * gray, (k, k))
    std     = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    return mean.astype(np.float32), normalize01(std)


def conservative_road_mask(bgr):
    h, w   = bgr.shape[:2]
    hsv    = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    sky = np.zeros((h, w), dtype=np.uint8)
    top_h = max(1, h // 2)
    sky[:top_h][(H[:top_h] >= 85) & (H[:top_h] <= 140) & (S[:top_h] <= 135) & (V[:top_h] >= 80)] = 1

    g   = bgr[:, :, 1].astype(np.int16)
    r   = bgr[:, :, 2].astype(np.int16)
    veg = ((H >= 25) & (H <= 95) & (S >= 35) & (g > r + 5)).astype(np.uint8)

    road = (1 - np.clip(sky + veg, 0, 1)).astype(np.uint8)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    road = cv2.morphologyEx(road * 255, cv2.MORPH_CLOSE, k, iterations=2)
    road = cv2.morphologyEx(road,       cv2.MORPH_OPEN,  k, iterations=1)
    road = (road > 0).astype(np.uint8)

    n, labels, stats, cent = cv2.connectedComponentsWithStats(road, 8)
    if n <= 1:
        fallback = np.zeros((h, w), dtype=np.uint8)
        fallback[h // 3:] = 1
        return fallback

    best, best_score = 1, -1
    for lab in range(1, n):
        score = stats[lab, cv2.CC_STAT_AREA] * (0.4 + cent[lab][1] / max(h, 1))
        if score > best_score:
            best_score, best = score, lab
    road               = (labels == best).astype(np.uint8)
    road[:int(h*0.06)] = 0
    return road


GABOR_KERNELS = [
    cv2.getGaborKernel((15, 15), 3.0, theta, 8.0, 0.5, 0, ktype=cv2.CV_32F)
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
]


def build_feature_map(bgr):
    h, w   = bgr.shape[:2]
    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv    = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab    = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_u8= (gray * 255).astype(np.uint8)

    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_u8).astype(np.float32) / 255.0
    illum   = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    illum_n = normalize01(gray / (illum + 1e-4))

    sx       = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy       = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = normalize01(np.sqrt(sx * sx + sy * sy))
    grad_ang = (np.arctan2(sy, sx) + np.pi) / (2 * np.pi)
    lap      = normalize01(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)))

    mean7,  std7  = local_mean_std(gray, 7)
    mean15, std15 = local_mean_std(gray, 15)

    blackhat_feats = [
        normalize01(cv2.morphologyEx(
            gray_u8, cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        ).astype(np.float32))
        for ks in (15, 31, 61)
    ]

    lbp     = local_binary_pattern(gray_u8, P=8, R=1, method="uniform").astype(np.float32)
    lbp_max = lbp.max()
    if lbp_max > 0:
        lbp /= lbp_max

    gabor_feats = [
        normalize01(np.abs(cv2.filter2D(gray, cv2.CV_32F, kern)))
        for kern in GABOR_KERNELS
    ]

    ys, xs       = np.mgrid[0:h, 0:w]
    x_norm       = xs.astype(np.float32) / max(w - 1, 1)
    y_norm       = ys.astype(np.float32) / max(h - 1, 1)
    bottom_prior = y_norm ** 2
    center_x     = np.abs(x_norm - 0.5)
    road         = conservative_road_mask(bgr).astype(np.float32)

    H = hsv[:, :, 0] / 179.0;  S = hsv[:, :, 1] / 255.0;  V = hsv[:, :, 2] / 255.0
    L = lab[:, :, 0] / 255.0;  A = lab[:, :, 1] / 255.0;  B = lab[:, :, 2] / 255.0

    low_sat       = 1.0 - S
    dark          = 1.0 - V
    wet_like      = normalize01(low_sat * (0.5 * V + 0.5 * (1.0 - std15)))
    shadow_like   = normalize01(dark * low_sat * (1.0 - grad_mag))
    dark_edge     = normalize01(dark * grad_mag)
    specular_like = normalize01((V > 0.75).astype(np.float32) * low_sat * (1.0 - S))

    maps = [
        rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2],
        H, S, V, L, A, B,
        gray, clahe, illum_n,
        grad_mag, grad_ang, lap,
        mean7, std7, mean15, std15,
        lbp,
        x_norm, y_norm, bottom_prior, center_x,
        road,
        wet_like, shadow_like, dark_edge, specular_like,
    ]
    maps.extend(blackhat_feats)
    maps.extend(gabor_feats)
    feat = np.stack(maps, axis=-1).astype(np.float32)

    if feat.shape[-1] != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Feature mismatch: computed {feat.shape[-1]}, config expects {len(FEATURE_NAMES)}."
        )
    return feat


def positive_proba(model, X, batch_size=300_000):
    out = np.zeros(X.shape[0], dtype=np.float32)
    for start in range(0, X.shape[0], batch_size):
        end  = min(start + batch_size, X.shape[0])
        raw  = np.asarray(model.predict_proba(X[start:end]), dtype=np.float32)
        if raw.ndim == 1:
            out[start:end] = raw
        elif hasattr(model, "classes_") and 1 in np.asarray(model.classes_).astype(int):
            idx = int(np.where(np.asarray(model.classes_).astype(int) == 1)[0][0])
            out[start:end] = raw[:, idx]
        else:
            out[start:end] = raw[:, -1]
    return out


def predict_probability_map(bgr, model, work_size):
    bgr_r    = cv2.resize(bgr, (work_size, work_size), interpolation=cv2.INTER_AREA)
    feat     = build_feature_map(bgr_r)
    X        = feat.reshape(-1, feat.shape[-1]).astype(np.float32)
    t0       = time.perf_counter()
    prob     = positive_proba(model, X).reshape(work_size, work_size)
    model_ms = (time.perf_counter() - t0) * 1000
    road_ch  = feat[:, :, FEATURE_NAMES.index("road_mask")]
    prob     = prob * (road_ch > 0.5)
    return prob.astype(np.float32), model_ms


# ===========================================================================
# POST-PROCESSING
# ===========================================================================

def remove_small_components(mask, min_area=250, keep_largest=False):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out   = np.zeros_like(mask, dtype=np.uint8)
    comps = [(int(stats[lab, cv2.CC_STAT_AREA]), lab)
             for lab in range(1, n) if stats[lab, cv2.CC_STAT_AREA] >= min_area]
    if keep_largest and comps:
        comps = [max(comps)]
    for _, lab in comps:
        out[labels == lab] = 1
    return out


def fill_holes(mask):
    mask_u8 = (mask > 0).astype(np.uint8)
    h, w    = mask_u8.shape
    flood   = (mask_u8 * 255).astype(np.uint8)
    temp    = flood.copy()
    cv2.floodFill(temp, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return (cv2.bitwise_or(flood, cv2.bitwise_not(temp)) > 0).astype(np.uint8)


def postprocess_probability(prob, threshold, min_area, close_k, open_k, fill,
                             max_area_ratio=0.22, keep_largest=False):
    th = float(threshold)
    while th <= 0.93:
        if (prob >= th).mean() <= max_area_ratio:
            break
        th += 0.04

    mask = (prob >= th).astype(np.uint8)
    if close_k and close_k > 1:
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_k), int(close_k)))
        mask = (cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE, k) > 0).astype(np.uint8)
    if open_k and open_k > 1:
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_k), int(open_k)))
        mask = (cv2.morphologyEx(mask * 255, cv2.MORPH_OPEN, k) > 0).astype(np.uint8)
    if fill:
        mask = fill_holes(mask)
    mask = remove_small_components(mask, min_area=int(min_area), keep_largest=keep_largest)
    return mask.astype(np.uint8), th


# ===========================================================================
# VISUALISATION HELPERS
# ===========================================================================

def make_overlay(bgr, mask, color_bgr=(42, 78, 214), alpha=0.55):
    out      = bgr.copy()
    color    = np.zeros_like(bgr)
    color[:] = color_bgr
    px       = mask > 0
    out[px]  = (alpha * color[px] + (1 - alpha) * bgr[px]).astype(np.uint8)
    return out


def bgr_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compute_binary_metrics(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt   = (gt > 127).astype(np.uint8) if gt.max() > 1 else (gt > 0).astype(np.uint8)
    if pred.shape != gt.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
    eps = 1e-8
    tp  = float(((pred == 1) & (gt == 1)).sum())
    tn  = float(((pred == 0) & (gt == 0)).sum())
    fp  = float(((pred == 1) & (gt == 0)).sum())
    fn  = float(((pred == 0) & (gt == 1)).sum())
    iou = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    dice   = 2 * tp / (2 * tp + fp + fn + eps)
    prec   = tp / (tp + fp + eps)
    rec    = tp / (tp + fn + eps)
    f1bg   = 2 * tn / (2 * tn + fp + fn + eps)
    return {
        "IoU Pothole":    round(iou,                       4),
        "mIoU":           round((iou + iou_bg) / 2,        4),
        "Dice":           round(dice,                       4),
        "Pixel Accuracy": round((tp+tn)/(tp+tn+fp+fn+eps), 4),
        "Precision":      round(prec,                       4),
        "Recall":         round(rec,                        4),
        "Macro F1":       round((dice + f1bg) / 2,         4),
    }


# ===========================================================================
# DESIGN SYSTEM — CSS
# ===========================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
/* ---- Variables ---- */
:root {
  --bg:         #0B0E14;
  --s1:         #111520;
  --s2:         #18202E;
  --s3:         #1F2A3C;
  --border:     #2A3347;
  --border-hi:  #3D4F6A;
  --accent:     #D64E2A;
  --accent-lo:  rgba(214,78,42,.10);
  --accent-md:  rgba(214,78,42,.22);
  --text:       #DDD8CE;
  --muted:      #8A95A8;
  --dim:        #4E5C72;
  --green:      #3DA05C;
  --green-lo:   rgba(61,160,92,.12);
  --yellow:     #C8942A;
  --r:          6px;
  --r-lg:       12px;
  --display:    'Space Grotesk', system-ui, sans-serif;
  --mono:       'JetBrains Mono', 'Fira Mono', monospace;
}

/* ---- Global reset ---- */
#MainMenu, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.stApp { background: var(--bg) !important; }
.block-container {
  padding: 1.5rem 2.5rem 4rem !important;
  max-width: 1340px !important;
}
h1,h2,h3,h4 { font-family: var(--display) !important; color: var(--text) !important; }
p, li, span { color: var(--text); }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
  background: var(--s1) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* Sidebar text */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCheckbox label {
  color: var(--text) !important;
  font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
  font-size: 0.7rem !important;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--dim) !important;
  margin-top: 1.2rem;
}
[data-testid="stSidebar"] [data-testid="stCaption"] {
  color: var(--dim) !important;
  font-size: 0.75rem !important;
}

/* Sidebar navigation items */
[data-testid="stSidebarNavItems"] a {
  color: var(--muted) !important;
  font-family: var(--display) !important;
  font-size: 0.875rem !important;
  font-weight: 500;
  border-radius: var(--r) !important;
  transition: color 0.15s, background 0.15s;
}
[data-testid="stSidebarNavItems"] a:hover {
  color: var(--text) !important;
  background: var(--s2) !important;
}
[data-testid="stSidebarNavItems"] [aria-selected="true"] a {
  color: var(--text) !important;
  background: var(--s3) !important;
  border-left: 2px solid var(--accent) !important;
}

/* ---- Dividers ---- */
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

/* ---- Buttons ---- */
.stButton > button {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--r) !important;
  font-family: var(--display) !important;
  font-weight: 600 !important;
  font-size: 0.875rem !important;
  padding: 0.55rem 1.25rem !important;
  letter-spacing: 0.01em;
  transition: filter 0.15s, transform 0.1s !important;
  box-shadow: 0 2px 12px rgba(214,78,42,.25) !important;
}
.stButton > button:hover {
  filter: brightness(1.12) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ---- Download button ---- */
[data-testid="stDownloadButton"] > button {
  background: var(--s2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  font-family: var(--display) !important;
  font-size: 0.83rem !important;
  font-weight: 500 !important;
  box-shadow: none !important;
}
[data-testid="stDownloadButton"] > button:hover {
  border-color: var(--border-hi) !important;
  background: var(--s3) !important;
  filter: none !important;
  transform: none !important;
}

/* ---- File uploader ---- */
[data-testid="stFileUploader"] {
  background: var(--s1) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: var(--r-lg) !important;
  transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--border-hi) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploader"] label {
  color: var(--muted) !important;
  font-size: 0.85rem !important;
}
[data-testid="stFileUploader"] small { color: var(--dim) !important; font-size: 0.75rem !important; }

/* ---- Sliders ---- */
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--text) !important; font-family: var(--mono) !important; font-size: 0.78rem !important; }
[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; border: 2px solid var(--accent) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [class*="TrackFill"] { background: var(--accent) !important; }

/* ---- Select / Selectbox ---- */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectSlider"] > div > div {
  background: var(--s2) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}

/* ---- Radio ---- */
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: var(--text) !important; font-size: 0.85rem !important; }
[data-testid="stRadio"] [role="radiogroup"] label div:first-child { border-color: var(--border-hi) !important; }
[data-testid="stRadio"] [aria-checked="true"] div:first-child { background: var(--accent) !important; border-color: var(--accent) !important; }

/* ---- Checkbox ---- */
[data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] p { color: var(--text) !important; font-size: 0.85rem !important; }
[data-testid="stCheckbox"] input:checked + div { background: var(--s3) !important; border-color: var(--border-hi) !important; }
[data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] { background: var(--s3) !important; border-color: var(--border-hi) !important; }
[data-baseweb="checkbox"] [data-checked="true"] { background: var(--s3) !important; border-color: var(--border-hi) !important; }

/* ---- Expander ---- */
[data-testid="stExpander"] {
  background: var(--s1) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
}
[data-testid="stExpander"] summary { color: var(--text) !important; font-family: var(--display) !important; font-size: 0.875rem !important; font-weight: 500; }
[data-testid="stExpander"] summary:hover { color: var(--text) !important; }
[data-testid="stExpander"] svg { fill: var(--muted) !important; }

/* ---- Spinner ---- */
[data-testid="stSpinner"] { color: var(--muted) !important; }

/* ---- Info / warning boxes ---- */
[data-testid="stAlert"] {
  background: var(--s2) !important;
  border-radius: var(--r) !important;
}
[data-testid="stAlert"][data-baseweb="notification"] { border-left-color: var(--accent) !important; }

/* ---- Dataframe ---- */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: var(--r) !important; overflow: hidden; }
[data-testid="stDataFrame"] th { background: var(--s2) !important; color: var(--muted) !important; font-family: var(--display) !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stDataFrame"] td { color: var(--text) !important; font-family: var(--mono) !important; font-size: 0.82rem !important; }

/* ---- Captions ---- */
[data-testid="stCaptionContainer"] p,
[data-testid="stCaption"] { color: var(--dim) !important; font-size: 0.78rem !important; }

/* ---- Images ---- */
[data-testid="stImage"] img {
  border-radius: var(--r) !important;
  border: 1px solid var(--border);
}

/* ---- Expander inner content text ---- */
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
  color: var(--muted) !important;
  font-size: 0.875rem !important;
  line-height: 1.65 !important;
}

/* ---- File uploader inner zone ---- */
[data-testid="stFileUploaderDropzone"] {
  background: transparent !important;
  padding: 1rem !important;
}

/* ---- Reduce sidebar top padding ---- */
[data-testid="stSidebarContent"] { padding-top: 1rem !important; }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--s1); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-hi); }

/* ---- Containers with border ---- */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--s1) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-lg) !important;
  padding: 1rem !important;
}

/* ===============================================
   CUSTOM COMPONENTS
   =============================================== */

/* -- Hero -- */
.hero {
  position: relative;
  overflow: hidden;
  background: var(--s1);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 2.2rem 2.5rem 2rem;
  margin-bottom: 2rem;
}
/* Diagonal asphalt-hash texture — the signature element */
.hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background-image:
    repeating-linear-gradient(
      -50deg,
      transparent 0px, transparent 18px,
      rgba(255,255,255,0.018) 18px, rgba(255,255,255,0.018) 19px
    ),
    repeating-linear-gradient(
      40deg,
      transparent 0px, transparent 28px,
      rgba(255,255,255,0.010) 28px, rgba(255,255,255,0.010) 29px
    );
  pointer-events: none;
}
.hero-eyebrow {
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.75rem;
}
.hero h1 {
  font-family: var(--display) !important;
  font-size: 2.6rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.03em;
  color: var(--text) !important;
  line-height: 1.1;
  margin: 0 0 0.6rem !important;
}
.hero-sub {
  color: var(--muted) !important;
  font-size: 0.92rem !important;
  line-height: 1.65;
  max-width: 680px;
  margin: 0 0 1.2rem !important;
}
.hero-tags { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.hero-tag {
  display: inline-block;
  background: var(--s3);
  border: 1px solid var(--border);
  color: var(--muted);
  padding: 0.22rem 0.65rem;
  border-radius: 4px;
  font-family: var(--mono);
  font-size: 0.72rem;
  font-weight: 600;
}

/* -- Section header -- */
.sec-head {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin: 1.8rem 0 0.8rem;
}
.sec-num {
  font-family: var(--mono);
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--accent);
  background: var(--accent-lo);
  border: 1px solid var(--accent-md);
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  letter-spacing: 0.05em;
  flex-shrink: 0;
}
.sec-title {
  font-family: var(--display);
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--text);
}
.sec-desc {
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.55;
  margin: -0.3rem 0 0.9rem;
  padding-left: 0;
}

/* -- Metric cards -- */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.75rem;
  margin: 1rem 0;
}
.metric-card {
  background: var(--s1);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 0.9rem 1rem 0.8rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.18s;
}
.metric-card:hover { border-color: var(--border-hi); }
.metric-card::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  background: var(--accent);
  opacity: 0.35;
}
.metric-label {
  font-family: var(--display);
  font-size: 0.72rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--dim);
  margin-bottom: 0.35rem;
}
.metric-value {
  font-family: var(--mono);
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text);
  line-height: 1;
}
.metric-value.accent { color: var(--accent); }

/* -- Image label -- */
.img-label {
  font-family: var(--display);
  font-size: 0.78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--dim);
  margin-bottom: 0.4rem;
}

/* -- Status dot -- */
.status-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.83rem; margin: 0.3rem 0; color: var(--text); }
.dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot-green { background: var(--green); box-shadow: 0 0 0 3px var(--green-lo); }
.dot-red   { background: var(--accent); box-shadow: 0 0 0 3px var(--accent-lo); }
.status-detail { color: var(--dim); font-size: 0.75rem; font-family: var(--mono); }

/* -- Pipeline flow -- */
.flow {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.3rem;
  margin: 0.6rem 0 1rem;
}
.flow-step {
  background: var(--s2);
  border: 1px solid var(--border);
  color: var(--muted);
  padding: 0.25rem 0.6rem;
  border-radius: 4px;
  font-family: var(--mono);
  font-size: 0.75rem;
  font-weight: 600;
}
.flow-arrow { color: var(--dim); font-size: 0.75rem; }

/* -- Feature group card -- */
.feat-group {
  background: var(--s2);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 0.75rem 0.9rem 0.65rem;
  margin-bottom: 0.5rem;
}
.feat-group-title {
  font-family: var(--display);
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--accent);
  margin-bottom: 0.45rem;
}
.feat-tag {
  display: inline-block;
  background: var(--s3);
  border: 1px solid var(--border);
  color: var(--muted);
  padding: 0.18rem 0.45rem;
  border-radius: 4px;
  font-family: var(--mono);
  font-size: 0.71rem;
  margin: 0.15rem 0.15rem 0 0;
}

/* -- Heatmap colorscale legend -- */
.heatmap-legend { margin: 0 0 0.6rem; }
.heatmap-bar {
  height: 7px;
  border-radius: 4px;
  background: linear-gradient(90deg,
    #00007F 0%, #0000FF 12%, #007FFF 25%,
    #00FFFF 37%, #7FFF7F 50%,
    #FFFF00 63%, #FF7F00 75%,
    #FF0000 87%, #7F0000 100%
  );
  margin-bottom: 0.3rem;
}
.heatmap-labels {
  display: flex;
  justify-content: space-between;
  font-family: var(--mono);
  font-size: 0.68rem;
  color: var(--dim);
}

/* -- Info / highlight box -- */
.info-box {
  background: var(--s2);
  border-left: 3px solid var(--accent);
  border-radius: 0 var(--r) var(--r) 0;
  padding: 0.65rem 0.9rem;
  margin: 0.5rem 0;
  font-size: 0.85rem;
  color: var(--muted);
  line-height: 1.55;
}
.info-box strong { color: var(--text); }

.note-box {
  background: var(--green-lo);
  border-left: 3px solid var(--green);
  border-radius: 0 var(--r) var(--r) 0;
  padding: 0.6rem 0.9rem;
  margin: 0.5rem 0;
  font-size: 0.83rem;
  color: var(--muted);
}
.note-box strong { color: var(--green); }

/* -- Upload zone hint -- */
.upload-hint {
  font-size: 0.78rem;
  color: var(--dim);
  font-family: var(--mono);
  margin: 0.3rem 0 0.6rem;
}

/* -- Eval score row -- */
.score-grid {
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 0.5rem;
  margin: 0.75rem 0;
}
.score-card {
  background: var(--s2);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 0.6rem 0.5rem;
  text-align: center;
}
.score-name {
  font-family: var(--display);
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--dim);
  margin-bottom: 0.3rem;
}
.score-val {
  font-family: var(--mono);
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--text);
}
.score-val.hi { color: var(--green); }
.score-val.lo { color: var(--accent); }

/* -- Sidebar section label -- */
.sidebar-section {
  font-family: var(--mono);
  font-size: 0.65rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--dim);
  margin: 1.2rem 0 0.5rem;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid var(--border);
}

/* -- Separator -- */
.sep {
  height: 1px;
  background: var(--border);
  margin: 1.8rem 0;
}

/* -- Column image area -- */
.col-img-head {
  font-family: var(--mono);
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--dim);
  margin-bottom: 0.4rem;
}

/* responsive: collapse metric grid to 2-col on narrow screens */
@media (max-width: 900px) {
  .metric-grid { grid-template-columns: repeat(2, 1fr); }
  .score-grid  { grid-template-columns: repeat(3, 1fr); }
}

/* reduced motion */
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
</style>
"""


# ===========================================================================
# SHARED HTML HELPERS
# ===========================================================================

def render_hero():
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">BINUS University &nbsp;&middot;&nbsp; Semester 4 &nbsp;&middot;&nbsp; Computer Vision</div>
      <h1>Pothole<br>Detection</h1>
      <p class="hero-sub">
        Pixel-wise road damage segmentation using a 33-feature soft-probability ensemble.
        Trained on ARA&nbsp;7.0 with RDD2022 India hard negatives.
        No deep learning — pure classical machine learning.
      </p>
      <div class="hero-tags">
        <span class="hero-tag">LightGBM</span>
        <span class="hero-tag">XGBoost</span>
        <span class="hero-tag">CatBoost</span>
        <span class="hero-tag">33 features / pixel</span>
        <span class="hero-tag">Road mask</span>
        <span class="hero-tag">Soft-probability ensemble</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def sec(num, title, desc=""):
    desc_html = f'<p class="sec-desc">{desc}</p>' if desc else ""
    st.markdown(f"""
    <div class="sec-head">
      <span class="sec-num">{num}</span>
      <span class="sec-title">{title}</span>
    </div>
    {desc_html}
    """, unsafe_allow_html=True)


def metrics_row(items: list[tuple[str, str, bool]]):
    """items = list of (label, value, use_accent_color)"""
    n     = len(items)
    cards = "".join(
        f'<div class="metric-card">'
        f'<div class="metric-label">{lbl}</div>'
        f'<div class="metric-value{" accent" if hi else ""}">{val}</div>'
        f'</div>'
        for lbl, val, hi in items
    )
    st.markdown(
        f'<div class="metric-grid" style="grid-template-columns:repeat({n},1fr)">{cards}</div>',
        unsafe_allow_html=True,
    )


def flow(steps: list[str]):
    parts = []
    for i, s in enumerate(steps):
        parts.append(f'<span class="flow-step">{s}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="flow-arrow">&rarr;</span>')
    st.markdown(f'<div class="flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def status_dot(label, ok, detail=""):
    cls   = "dot-green" if ok else "dot-red"
    extra = f'<span class="status-detail">&nbsp;{detail}</span>' if detail else ""
    st.markdown(
        f'<div class="status-row"><span class="dot {cls}"></span>{label}{extra}</div>',
        unsafe_allow_html=True,
    )


def sep():
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)


def info_box(text: str):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def note_box(text: str):
    st.markdown(f'<div class="note-box">{text}</div>', unsafe_allow_html=True)


def col_img_head(title: str):
    st.markdown(f'<div class="col-img-head">{title}</div>', unsafe_allow_html=True)


def score_row(metrics: dict[str, float]):
    def cls(v):
        return "hi" if v >= 0.65 else ("lo" if v < 0.40 else "")
    cards = "".join(
        f'<div class="score-card">'
        f'<div class="score-name">{k}</div>'
        f'<div class="score-val {cls(v)}">{v:.4f}</div>'
        f'</div>'
        for k, v in metrics.items()
    )
    st.markdown(f'<div class="score-grid">{cards}</div>', unsafe_allow_html=True)


def heatmap_legend():
    st.markdown("""
    <div class="heatmap-legend">
      <div class="heatmap-bar"></div>
      <div class="heatmap-labels"><span>Low probability</span><span>High probability</span></div>
    </div>
    """, unsafe_allow_html=True)


def sidebar_section(title: str):
    st.markdown(f'<div class="sidebar-section">{title}</div>', unsafe_allow_html=True)


# ===========================================================================
# SIDEBAR
# ===========================================================================

def render_sidebar(fast_ok: bool, acc_ok: bool):
    with st.sidebar:
        sidebar_section("Status")
        status_dot("Real-time model", fast_ok, "ready" if fast_ok else "not found")
        status_dot("Accuracy model",  acc_ok,  "ready" if acc_ok  else "not found")

        sidebar_section("Inference")
        mode = st.radio(
            "Mode",
            ["Real-time", "Accuracy"],
            index=0,
            help="Real-time uses the fast single model. Accuracy uses the full ensemble.",
        )
        work_size = st.selectbox(
            "Processing resolution",
            [224, 256, 320, 384],
            index=[224, 256, 320, 384].index(DEFAULT_WORK_SIZE)
                  if DEFAULT_WORK_SIZE in [224, 256, 320, 384] else 1,
            help="Image is resized to this square before feature extraction. Larger = slower but cleaner mask.",
        )
        st.caption("256 px — faster  /  384 px — cleaner mask")

        sidebar_section("Post-processing")
        threshold      = st.slider("Probability threshold", 0.30, 0.95, DEFAULT_THRESHOLD, 0.01,
                                   help="Pixels above this value are classified as pothole.")
        min_area       = st.slider("Min component area (px)", 50, 3000, DEFAULT_MIN_AREA, 50,
                                   help="Connected components smaller than this are removed as noise.")
        close_k        = st.select_slider("Morphological close", options=[0, 3, 5, 7, 9, 11],
                                          value=DEFAULT_CLOSE_K if DEFAULT_CLOSE_K in [0,3,5,7,9,11] else 5,
                                          help="Closes small gaps inside pothole regions.")
        open_k         = st.select_slider("Morphological open",  options=[0, 3, 5, 7],
                                          value=DEFAULT_OPEN_K  if DEFAULT_OPEN_K  in [0,3,5,7] else 3,
                                          help="Removes isolated noise outside pothole regions.")
        fill           = st.checkbox("Fill interior holes", value=DEFAULT_FILL,
                                     help="Flood-fill enclosed voids inside detected potholes.")
        max_area_ratio = st.slider("Max area ratio", 0.03, 0.60, DEFAULT_MAX_AREA, 0.01,
                                   help="If predicted area exceeds this fraction of the image, threshold is auto-raised.")
        keep_largest   = st.checkbox("Keep largest region only", value=False,
                                     help="Discard all but the largest connected component.")

        sidebar_section("Artifact")
        st.caption(f"Path: {ARTIFACT_DIR.name}")
        st.caption(MODEL_NOTE)

    return mode, int(work_size), threshold, min_area, close_k, open_k, fill, max_area_ratio, keep_largest


# ===========================================================================
# PAGE — DETECTION
# ===========================================================================

def page_detect(fast_model, acc_model, params):
    mode, work_size, threshold, min_area, close_k, open_k, fill, max_area_ratio, keep_largest = params
    model = fast_model if mode == "Real-time" else acc_model

    # -- 01 Upload -----------------------------------------------------------
    sec("01", "Upload road image", "JPG, PNG, BMP, WebP, or TIFF — any resolution.")
    uploaded = st.file_uploader(
        "Drop image here or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        info_box("Upload a road image to begin. Switch to <strong>Model Info</strong> in the sidebar to learn how the pipeline works.")
        return

    bgr = read_uploaded_image(uploaded)
    if bgr is None:
        st.error("Could not read the file. Try a different image (JPG or PNG).")
        return

    h, w  = bgr.shape[:2]
    c_img, c_meta = st.columns([5, 2])
    with c_img:
        col_img_head("Uploaded image")
        st.image(bgr_to_rgb(bgr), use_container_width=True)
    with c_meta:
        st.markdown("<br>", unsafe_allow_html=True)
        metrics_row([
            ("Width",       f"{w} px",    False),
            ("Height",      f"{h} px",    False),
        ])
        metrics_row([
            ("Mode",        mode,         False),
            ("Proc. size",  f"{work_size} px", False),
        ])

    # -- 02 Ground truth (optional) ------------------------------------------
    sep()
    sec("02", "Ground truth mask", "Optional. Grayscale PNG — white = pothole, black = road. Upload to compute evaluation metrics.")
    uploaded_gt = st.file_uploader(
        "Drop ground truth mask here",
        type=["png", "jpg", "jpeg"],
        key="gt_uploader",
        label_visibility="collapsed",
    )

    gt_bin = None
    if uploaded_gt is not None:
        gt_raw = cv2.imdecode(
            np.frombuffer(uploaded_gt.getvalue(), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        if gt_raw is None:
            st.error("Could not read the mask. Confirm it is a valid grayscale image.")
        else:
            if gt_raw.shape != (h, w):
                gt_raw = cv2.resize(gt_raw, (w, h), interpolation=cv2.INTER_NEAREST)
            gt_candidate = (gt_raw > 127).astype(np.uint8) * 255
            n_px         = int((gt_candidate > 0).sum())

            g1, g2 = st.columns([4, 2])
            with g1:
                col_img_head("Ground truth overlay")
                gt_ov = make_overlay(bgr, gt_candidate, color_bgr=(61, 160, 92), alpha=0.5)
                st.image(bgr_to_rgb(gt_ov), use_container_width=True)
            with g2:
                st.markdown("<br>", unsafe_allow_html=True)
                metrics_row([
                    ("Pothole px",  f"{n_px:,}",                         False),
                    ("Coverage",    f"{n_px / gt_candidate.size * 100:.2f}%", n_px > 0),
                ])

            if n_px == 0:
                info_box("The mask appears empty — no white pixels found. Check the file format.")
            else:
                gt_bin = gt_candidate
                note_box("<strong>Ground truth loaded.</strong> Metrics will appear after detection.")

    # -- 03 Run detection ----------------------------------------------------
    sep()
    sec("03", "Run detection")

    if model is None:
        st.error(f"The {mode} model is not available. Check the artifact folder.")
        return

    if not st.button("Run Detection", type="primary", use_container_width=True):
        return

    with st.spinner("Extracting 33-channel feature map and running ensemble inference…"):
        t_total                = time.perf_counter()
        prob, model_ms         = predict_probability_map(bgr, model, work_size)
        mask_small, used_th    = postprocess_probability(
            prob, threshold=threshold, min_area=min_area,
            close_k=close_k, open_k=open_k, fill=fill,
            max_area_ratio=max_area_ratio, keep_largest=keep_largest,
        )
        mask      = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        total_ms  = (time.perf_counter() - t_total) * 1000

    area_pct = float((mask > 0).mean() * 100)

    metrics_row([
        ("Detected area",   f"{area_pct:.2f}%",  area_pct > 0),
        ("Total time",      f"{total_ms:.0f} ms", False),
        ("Model time",      f"{model_ms:.0f} ms", False),
        ("Threshold used",  f"{used_th:.2f}",     False),
    ])

    sep()

    # Visual results
    overlay  = make_overlay(bgr, mask)
    prob_vis = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)

    c1, c2, c3 = st.columns(3)
    with c1:
        col_img_head("Original")
        st.image(bgr_to_rgb(bgr), use_container_width=True)
    with c2:
        col_img_head("Predicted mask")
        st.image(mask * 255, clamp=True, use_container_width=True)
    with c3:
        col_img_head("Overlay — red = pothole")
        st.image(bgr_to_rgb(overlay), use_container_width=True)

    with st.expander("Probability heatmap (JET colorscale)"):
        heatmap_legend()
        st.image(bgr_to_rgb(prob_vis), use_container_width=True)

    ok, buf = cv2.imencode(".png", mask * 255)
    if ok:
        st.download_button(
            "Download predicted mask",
            data=buf.tobytes(),
            file_name="predicted_mask.png",
            mime="image/png",
        )

    # -- 04 Evaluation (only when GT available) ------------------------------
    if gt_bin is not None:
        sep()
        sec("04", "Evaluation against ground truth")
        m = compute_binary_metrics(mask, gt_bin)
        score_row(m)

        col_t, col_c = st.columns([1, 1])
        with col_t:
            df = (pd.DataFrame([m]).T.reset_index()
                    .rename(columns={"index": "Metric", 0: "Value"}))
            df["Value"] = df["Value"].map(lambda x: f"{x:.4f}")
            st.dataframe(df, use_container_width=True, hide_index=True)
        with col_c:
            st.bar_chart(
                pd.DataFrame({"Score": list(m.values())}, index=list(m.keys())),
                color="#D64E2A",
                horizontal=True,
            )
    else:
        info_box("Upload a ground truth mask in Step 02 to see evaluation metrics — IoU, Dice, Precision, Recall, and more.")


# ===========================================================================
# PAGE — MODEL INFO
# ===========================================================================

def page_info():
    sec("A", "Feature extraction pipeline",
        f"Every pixel becomes a {len(FEATURE_NAMES)}-dimensional vector before classification.")

    flow([
        "BGR input", f"Resize to {DEFAULT_WORK_SIZE}px²",
        "33-channel feature map", "Road mask filter",
        "Ensemble predict_proba", "Threshold + post-processing",
        "Segmentation mask",
    ])

    feature_groups = [
        ("Color — RGB",         ["R", "G", "B"]),
        ("Color — HSV",         ["Hue", "Saturation", "Value"]),
        ("Color — CIE L*a*b*",  ["L*", "a*", "b*"]),
        ("Intensity",           ["Grayscale", "CLAHE-equalized", "Illumination-normalized"]),
        ("Gradient & edge",     ["Sobel magnitude", "Sobel angle", "Laplacian abs"]),
        ("Local statistics",    ["Mean k=7", "Std k=7", "Mean k=15", "Std k=15"]),
        ("Texture",             ["LBP uniform", "Blackhat k=15", "Blackhat k=31", "Blackhat k=61"]),
        ("Gabor filters",       ["theta=0deg", "theta=45deg", "theta=90deg", "theta=135deg"]),
        ("Spatial priors",      ["x-position", "y-position", "Bottom prior y^2", "Center-x distance"]),
        ("Scene context",       ["Road mask", "Wet-like", "Shadow-like", "Dark edge", "Specular-like"]),
    ]

    cols = st.columns(3)
    for i, (title, tags) in enumerate(feature_groups):
        with cols[i % 3]:
            tags_html = "".join(f'<span class="feat-tag">{t}</span>' for t in tags)
            st.markdown(
                f'<div class="feat-group">'
                f'<div class="feat-group-title">{title}</div>'
                f'{tags_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

    sep()
    sec("B", "Ensemble architecture",
        "Three gradient-boosted classifiers vote on each pixel's probability via weighted average.")

    c_left, c_right = st.columns(2)
    with c_left:
        flow(["Feature vector (33d)", "LightGBM", "XGBoost", "CatBoost", "Weighted avg", "Threshold"])
        info_box(
            "<strong>Real-time mode</strong> — uses the fast single model (typically LightGBM) "
            "for sub-second latency on 256&nbsp;px input.<br><br>"
            "<strong>Accuracy mode</strong> — uses the full ensemble for the highest mask quality."
        )
    with c_right:
        st.markdown("""
**Why ensemble?**

- LightGBM and XGBoost excel on gradient and texture features
- CatBoost handles categorical-like color patterns well
- Averaging probabilities cancels each model's systematic errors
- Better-calibrated confidence makes the threshold more interpretable
        """)

    sep()
    sec("C", "Post-processing steps",
        "Raw probability maps are converted into clean binary masks through six ordered operations.")

    steps = [
        ("Auto-threshold adjustment",
         "If the predicted area exceeds the max area ratio, the threshold is raised by 0.04 "
         "repeatedly until the area is within bound. Prevents the model from over-labelling "
         "bright or overexposed frames."),
        ("Morphological closing",
         "Fills small internal gaps within pothole blobs. "
         "Controlled by the <em>close kernel</em> size parameter."),
        ("Morphological opening",
         "Removes isolated noise pixels outside pothole regions. "
         "Controlled by the <em>open kernel</em> size parameter."),
        ("Hole filling",
         "Optional flood-fill to close enclosed voids inside large pothole detections. "
         "Enabled with the <em>Fill interior holes</em> checkbox."),
        ("Small component removal",
         "Any connected component with area below <em>min component area</em> is discarded. "
         "Eliminates speckle and fragmented predictions."),
        ("Upscale to original resolution",
         "The binary mask produced at the working resolution is nearest-neighbour upsampled "
         "back to the original image dimensions before display and download."),
    ]

    for title, desc in steps:
        with st.expander(title):
            st.markdown(f'<span style="color:var(--muted);font-size:0.875rem;line-height:1.6">{desc}</span>',
                        unsafe_allow_html=True)

    sep()
    sec("D", "Road segmentation",
        "A conservative road mask restricts predictions to road pixels only, reducing false positives "
        "from sky, vegetation, and buildings.")

    flow([
        "HSV conversion",
        "Sky mask — upper half",
        "Vegetation mask — green hue",
        "Invert to road candidate",
        "Morphological clean-up",
        "Largest bottom-weighted component",
    ])
    info_box(
        "<strong>Fallback:</strong> if no valid road component is found — for example in "
        "images with minimal colour contrast — the bottom two-thirds of the frame is used "
        "as the road region."
    )

    sep()
    sec("E", "Heatmap colorscale reference")
    heatmap_legend()
    info_box(
        "The JET colormap maps model confidence to colour. "
        "<strong>Blue</strong> pixels have near-zero pothole probability. "
        "<strong>Red</strong> pixels have high pothole probability. "
        "The final binary mask thresholds this map at the chosen probability threshold."
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    st.set_page_config(
        page_title="Pothole Detection",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    try:
        fast_model, acc_model = load_models()
        fast_ok = fast_model is not None
        acc_ok  = acc_model  is not None
    except Exception as e:
        st.error(str(e))
        st.stop()
        return

    render_hero()

    params = render_sidebar(fast_ok, acc_ok)

    pages = [
        st.Page(
            lambda p=params: page_detect(fast_model, acc_model, p),
            title="Detection",
            icon=":material/search:",
            default=True,
        ),
        st.Page(page_info, title="Model Info", icon=":material/menu_book:"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
