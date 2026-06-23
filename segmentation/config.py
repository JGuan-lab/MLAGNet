"""Centralized path configuration. Default values are from the original experiment environment; override via this file or environment variables."""
import os

DATA_ROOT  = os.environ.get("MLAG_DATA_ROOT", "/home/yons/data/Weiyubing")
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.environ.get("MLAG_SEG_MODEL", "/home/yons/data/Weiyubing/Ad/ours")

# Dataset paths
ANI_TRAIN_IMG    = os.path.join(DATA_ROOT, "fengedata/anigiography/training/img")
ANI_TRAIN_LABEL  = os.path.join(DATA_ROOT, "fengedata/anigiography/training/labelcol")
ANI_TEST_IMG     = os.path.join(DATA_ROOT, "fengedata/anigiography/test/img")
ANI_TEST_LABEL   = os.path.join(DATA_ROOT, "fengedata/anigiography/test/labelcol")

CRACK_TRAIN_IMG   = os.path.join(DATA_ROOT, "fengedata/crack/training/img")
CRACK_TRAIN_LABEL = os.path.join(DATA_ROOT, "fengedata/crack/training/labelcol")
CRACK_TEST_IMG    = os.path.join(DATA_ROOT, "fengedata/crack/test/img")
CRACK_TEST_LABEL  = os.path.join(DATA_ROOT, "fengedata/crack/test/labelcol")

RETINA_TRAIN_IMG   = os.path.join(DATA_ROOT, "fengedata/retinaplus/training/img")
RETINA_TRAIN_LABEL = os.path.join(DATA_ROOT, "fengedata/retinaplus/training/labelcol")
RETINA_TEST_IMG    = os.path.join(DATA_ROOT, "fengedata/retinaplus/test/img")
RETINA_TEST_LABEL  = os.path.join(DATA_ROOT, "fengedata/retinaplus/test/labelcol")

# Model save dirs
ANI_MODEL_DIR   = os.path.join(MODEL_ROOT, "ani")
CRACK_MODEL_DIR = os.path.join(MODEL_ROOT, "crack")
ABI_MODEL_DIR   = os.path.join(MODEL_ROOT, "abi")

# Output dirs (relative to module dir, replacing /home/Yb/uureal/fenge/)
LOG_DIR    = os.path.join(MODULE_DIR, "logs")
RESULT_DIR = os.path.join(MODULE_DIR, "result")
MID_DIR    = os.path.join(MODULE_DIR, "midresult")

ANI_LOG_DIR    = os.path.join(LOG_DIR, "ani")
CRACK_LOG_DIR  = os.path.join(LOG_DIR, "crack")
RETINA_LOG_DIR = os.path.join(LOG_DIR, "retina")
ABI_LOG_DIR    = os.path.join(LOG_DIR, "abi")
NOLSA_LOG_DIR  = os.path.join(LOG_DIR, "NoLSA")

# trainretina.py specific: logs/retina/abi/NoLSA
RETINA_ABI_NOLSA_LOG_DIR = os.path.join(LOG_DIR, "retina", "abi", "NoLSA")

# Xiaorong ablation paths
XIAORONG_MODEL_NOGDFN = os.path.join(MODULE_DIR, "ablation", "model", "nogdfn")
XIAORONG_MODEL_NOLSA  = os.path.join(MODULE_DIR, "ablation", "model", "nolsa")
XIAORONG_MODEL_FULL   = os.path.join(MODULE_DIR, "ablation", "model", "full")
XIAORONG_RESULT_DIR   = os.path.join(MODULE_DIR, "ablation", "result")
XIAORONG_LOG_DIR      = os.path.join(MODULE_DIR, "ablation", "logs")

XIAORONG_RESULT_NOGDFN = os.path.join(XIAORONG_RESULT_DIR, "nogdfn")
XIAORONG_RESULT_NOLSA  = os.path.join(XIAORONG_RESULT_DIR, "nolsa")
XIAORONG_RESULT_FULL   = os.path.join(XIAORONG_RESULT_DIR, "full")

XIAORONG_LOG_NOGDFN = os.path.join(XIAORONG_LOG_DIR, "nogdfn")
XIAORONG_LOG_NOLSA  = os.path.join(XIAORONG_LOG_DIR, "nolsa")
XIAORONG_LOG_FULL   = os.path.join(XIAORONG_LOG_DIR, "full")

# Comparison model asset paths (used by test scripts)
COMMODEL_RETINAL_IMG   = os.path.join(MODULE_DIR, "commodel", "assets", "retinal", "img")
COMMODEL_RETINAL_LABEL = os.path.join(MODULE_DIR, "commodel", "assets", "retinal", "labelcol")
COMMODEL_CRACK_OURS    = os.path.join(MODULE_DIR, "commodel", "assets", "crack", "ours")
COMMODEL_CRACK_LOGS    = os.path.join(MODULE_DIR, "commodel", "assets", "crack", "logs")

# Specific trained model checkpoints
CRACK_BEST_MODEL = os.path.join(CRACK_MODEL_DIR, "model02_bestacc_0.9930419921875.pth")
XIAORONG_NOGDFN_BEST_MODEL = os.path.join(XIAORONG_MODEL_NOGDFN, "model02_bestacc_0.9843292236328125.pth")
XIAORONG_NOLSA_BEST_MODEL  = os.path.join(XIAORONG_MODEL_NOLSA,  "model02_bestacc_0.9854545593261719.pth")
XIAORONG_FULL_BEST_MODEL   = os.path.join(XIAORONG_MODEL_FULL,   "model02_bestacc_0.9749755859375.pth")

# Draw / visualization paths
DRAW_LOG_FILE    = os.path.join(RESULT_DIR, "train_log.txt")
DRAW_PLOT_OUTPUT = os.path.join(MODULE_DIR, "accuracy_plot.png")
