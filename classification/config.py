"""Centralized path configuration. Default values are absolute paths from the original experiment environment.
Edit this file or set environment variables to adapt to other machines.

Environment variable overrides:
  MLAG_DATA_ROOT   -- dataset root directory (default: /home/yons/data/Weiyubing)
  MLAG_CLS_MODEL   -- classification weights root (default: /home/yons/data/Weiyubing/fenleimodel)
  MLAG_OUT_ROOT    -- training output root (default: <classification module dir>, i.e., MODULE_DIR)
"""
import os

# ── Root directories ─────────────────────────────────────────────────────────
DATA_ROOT  = os.environ.get("MLAG_DATA_ROOT", "/home/yons/data/Weiyubing")
MODEL_ROOT = os.environ.get("MLAG_CLS_MODEL", "/home/yons/data/Weiyubing/fenleimodel")
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))  # classification/ directory

# ── Dataset paths ─────────────────────────────────────────────────────────────
# ISIC dermoscopy dataset
ISIC_TRAIN = os.path.join(DATA_ROOT, "IsIc/IsIc/train")
ISIC_VAL   = os.path.join(DATA_ROOT, "IsIc/IsIc/val")

# MHIST dataset
MHIST_TRAIN = os.path.join(DATA_ROOT, "fenleidata/mhsit/train")
MHIST_VAL   = os.path.join(DATA_ROOT, "fenleidata/mhsit/val")

# Histopathology dataset
HIST_TRAIN = os.path.join(DATA_ROOT, "fenleidata/histopathology/histopathology/train")
HIST_VAL   = os.path.join(DATA_ROOT, "fenleidata/histopathology/histopathology/valid")

# Chest X-ray dataset
XRAY_TRAIN = os.path.join(DATA_ROOT, "fenleidata/x/train")
XRAY_VAL   = os.path.join(DATA_ROOT, "fenleidata/x/val")

# ── Pretrained / comparison weight paths ─────────────────────────────────────
# best MHIST model loaded by test.py
TEST_MHIST_MODEL = os.path.join(MODEL_ROOT, "msh/best_model.pth")

# ViT weights loaded by redraw_vit_eval.py
VIT_HIST_MODEL  = os.path.join(MODEL_ROOT, "his_vit_best_model_fixed.pth")
VIT_MHIST_MODEL = os.path.join(MODEL_ROOT, "mhist_vit_best_model_fixed.pth")
VIT_XRAY_MODEL  = os.path.join(MODEL_ROOT, "x_vit_best_model_fixed.pth")
VIT_ISIC_MODEL  = os.path.join(MODEL_ROOT, "IsIc_vit_best_model_fixed.pth")

# ── Training output paths (relative to MODULE_DIR, works after cloning to any machine) ──
# Log directories
LOG_DIR    = os.path.join(MODULE_DIR, "logs")           # general log root (trainx/trainh/trainmhist)
LOG_DIR_AB = os.path.join(MODULE_DIR, "logs", "ab")     # ISIC/ablation logs

# Model save directories
MODEL_SAVE_NODE = os.path.join(MODULE_DIR, "model", "node")  # trainx/trainh/trainmhist
MODEL_SAVE_AB   = os.path.join(MODULE_DIR, "model", "ab")    # trainisic/ablation

# Confusion matrix image output
IMG_DIR_AB    = os.path.join(MODULE_DIR, "img", "ab")     # ISIC/ablation
IMG_DIR_X     = os.path.join(MODULE_DIR, "img", "x")      # X-ray
IMG_DIR_HIS   = os.path.join(MODULE_DIR, "img", "his")    # Histopathology
IMG_DIR_MHIST = os.path.join(MODULE_DIR, "img", "mhist")  # MHIST
IMG_DIR_MHIST_LOG = os.path.join(MODULE_DIR, "logs", "mhist")  # confusion matrix images for test.py

# ── redraw_vit_eval.py comparison model output ───────────────────────────────
VIT_OUT_HIST  = os.path.join(MODULE_DIR, "commodel", "vit", "img", "his_redraw")
VIT_OUT_MHIST = os.path.join(MODULE_DIR, "commodel", "vit", "img", "mhist_redraw")
VIT_OUT_XRAY  = os.path.join(MODULE_DIR, "commodel", "vit", "img", "x_redraw")
VIT_OUT_ISIC  = os.path.join(MODULE_DIR, "commodel", "vit", "img", "isic_redraw")
