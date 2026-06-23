"""Centralized path configuration. To reproduce, modify this file or set environment variables. Defaults are the original experiment absolute paths."""
import os

# ── Root directories ─────────────────────────────────────────────────────────
MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))  # this module's directory (not affected by env vars)
DATA_ROOT   = os.environ.get("MLAG_DATA_ROOT",   "/home/yons/data/Weiyubing")   # dataset root
OUTPUT_ROOT = os.environ.get("MLAG_SR_OUTPUT",   "/home/yons/data/Weiyubing/model")  # weights/output root
YB_ROOT     = os.environ.get("MLAG_YB_ROOT",     "/home/Yb/uureal")   # early experiment root (uureal)
YB_EUR      = os.environ.get("MLAG_YB_EUR",      "/home/Yb/EUR-GAN")  # early CT dataset root
YB_UU       = os.environ.get("MLAG_YB_UU",       "/home/Yb/uu")       # early training output root
WEIYB_ROOT  = os.environ.get("MLAG_WEIYB_ROOT",  "/home/weiyb/UU")    # alternative experiment environment root

# ── Training dataset paths ───────────────────────────────────────────────────
# trainVAL.py / trainNoWave.py (4x scale)
SR_TRAIN_GT_4X      = os.path.join(DATA_ROOT, "lungtotal")          # GT images (4x)
SR_TRAIN_LR_4X      = os.path.join(DATA_ROOT, "lungresize4")        # LR images (4x)

# tg.py (L1 model, different LR)
SR_TRAIN_GT_TG      = os.path.join(DATA_ROOT, "lungtotal")          # GT images (tg.py)
SR_TRAIN_LR_LUNG    = os.path.join(DATA_ROOT, "lunglow")            # LR images (lunglow)

# train.py (DIV2K)
SR_TRAIN_DIV2K      = os.path.join(DATA_ROOT, "DIV2")               # DIV2K dataset path

# processdata.py
SR_PROC_GT          = os.path.join(DATA_ROOT, "lungtotal")
SR_PROC_LQ          = os.path.join(DATA_ROOT, "lungtotal")

# small.py (Kaggle COVID downscaling)
SR_KAGGLE_TOTAL     = os.path.join(DATA_ROOT, "kagglecovid/sizetotal")   # original size
SR_KAGGLE_8X        = os.path.join(DATA_ROOT, "kagglecovid/sizetotal8")  # 8x downscaled
SR_KAGGLE_2X        = os.path.join(DATA_ROOT, "kagglecovid/sizetotal2")  # 2x downscaled

# topng.py (crack dataset jpg->png)
SR_CRACK_IMG_DIR    = os.path.join(DATA_ROOT, "fengedata/crack/test/images")
SR_CRACK_PNG_DIR    = os.path.join(DATA_ROOT, "fengedata/crack/test/imgs")

# testone.py (single image test)
SR_TESTONE_RESIZE   = os.path.join(DATA_ROOT, "lungresize")         # LR single-image directory
SR_TESTONE_GT_DIR   = os.path.join(DATA_ROOT, "lungretotal")        # GT single-image directory

# video.py / testfull*.py (qingtcrop dataset)
SR_QINGT_LR         = os.path.join(DATA_ROOT, "qingtcroplow")       # low resolution
SR_QINGT_HR         = os.path.join(DATA_ROOT, "qingtcroptrue")      # high resolution

# genvideo.py (GT sequence for video generation)
SR_GENVIDEO_GT      = os.path.join(DATA_ROOT, "qingtcroptrue")

# ── Validation set paths ─────────────────────────────────────────────────────
# trainVAL.py / trainNoWave.py (Ui/ctlow, Ui/cttrue)
SR_VAL_LR           = os.path.join(YB_ROOT, "Ui/ctlow")
SR_VAL_HR           = os.path.join(YB_ROOT, "Ui/cttrue")

# ── Model save paths ─────────────────────────────────────────────────────────
# trainVAL.py (4NoLSA experiment)
SR_MODEL_BEST_4NOLSA = os.path.join(OUTPUT_ROOT, "4NoLSA/")
SR_MODEL_G_4NOLSA    = os.path.join(OUTPUT_ROOT, "4NoLSA/")
SR_MODEL_D_4NOLSA    = os.path.join(OUTPUT_ROOT, "4NoLSAD/")

# trainNoWave.py (4StandardGAN experiment)
SR_MODEL_BEST_4STDGAN = os.path.join(OUTPUT_ROOT, "4StandardGAN/")
SR_MODEL_G_4STDGAN    = os.path.join(OUTPUT_ROOT, "4StandardGAN/")
SR_MODEL_D_4STDGAN    = os.path.join(OUTPUT_ROOT, "4StandardGAN_D/")

# tg.py (L1 model, 8 epochs)
SR_MODEL_G_L1_8     = os.path.join(OUTPUT_ROOT, "L1/8l1")

# train.py (early experiment)
SR_MODEL_G_OLD      = os.path.join(YB_UU, "model")
SR_MODEL_D_OLD      = os.path.join(YB_UU, "model")

# tg.py (early experiment, pathG hardcoded to uu)
SR_MODEL_G_TG       = os.path.join(YB_UU, "model/l1model")

# ── Model directories for evaluation ────────────────────────────────────────
SR_TEST_MODEL_4FULL = os.path.join(OUTPUT_ROOT, "4full")            # testfull.py
SR_TEST_MODEL_2FULL = os.path.join(OUTPUT_ROOT, "2full")            # testfull2.py
SR_TEST_MODEL_8FULL = os.path.join(OUTPUT_ROOT, "8full")            # testfull8.py
SR_TEST_MODEL_BEST  = os.path.join(YB_ROOT, "model/best")           # testssim.py

# ── Single-model test paths ──────────────────────────────────────────────────
SR_TESTONE_MODEL    = os.path.join(YB_ROOT, "finalmodel/4/netGl1loss_epoch_4_1780.pth")
SR_TESTONE_SR_OUT   = os.path.join(YB_ROOT, "result/4best/9999nolsa.png")

# video.py single model
SR_VIDEO_MODEL      = os.path.join(YB_ROOT, "model/gpm/best_14.0384558429318_netG.pth")

# ── Test data paths (testssim) ────────────────────────────────────────────────
SR_TEST_LR_SSIM     = os.path.join(YB_EUR, "ctlow")
SR_TEST_HR_SSIM     = os.path.join(YB_EUR, "true")

# ── Temporary SR result directories ─────────────────────────────────────────
SR_TMP_RESULTS_4    = "/tmp/sr_results2"                             # testfull.py (original commented value)
SR_TMP_RESULTS      = os.path.join(DATA_ROOT, "tmp")                # testfull.py (actually used)
SR_TMP_RESULTS_2    = "/tmp/sr_results2"                             # testfull2.py
SR_TMP_RESULTS_8    = "/tmp/sr_results8"                             # testfull8.py
SR_TMP_RESULTS_SSIM = "/tmp/sr_results"                              # testssim.py

# video.py dynamic paths (assembled by script using model name; this is the prefix)
SR_VIDEO_TMP_PREFIX = os.path.join(YB_ROOT, "result/tmp")
SR_VIDEO_OUT_PREFIX = os.path.join(YB_ROOT, "result/video")

# ── Result txt files ─────────────────────────────────────────────────────────
SR_RESULT_ERROR_TXT        = os.path.join(YB_ROOT, "result/txt/error.txt")
SR_RESULT_TXT_4TEST        = os.path.join(YB_ROOT, "result/txt/4test0119.txt")
SR_RESULT_TXT_2COVID       = os.path.join(YB_ROOT, "result/txt/2covid_kaggle.txt")
SR_RESULT_TXT_8COVID       = os.path.join(YB_ROOT, "result/txt/8kaggel_covid.txt")
SR_RESULT_TXT_BESTPHOTO    = os.path.join(YB_ROOT, "result/txt/bestphoto_evaluation_results.txt")
SR_RESULT_TXT_VIDEO        = os.path.join(YB_ROOT, "result/txt/single_model_evaluation_results.txt")
SR_RESULT_TXT_2FULL_EVA    = os.path.join(YB_ROOT, "result/txt/2full_eva_results.txt")    # modelpostprocess.py input
SR_RESULT_TXT_2FULL_FINAL  = os.path.join(YB_ROOT, "result/txt/2full_final_results.txt")  # modelpostprocess.py output
SR_RESULT_TXT_RANK         = os.path.join(YB_ROOT, "result/txt/4kaggle_covid.txt")         # rank.py

# deletemodel.py
SR_DELETE_TXT              = os.path.join(YB_ROOT, "result/txt/2full_final_results.txt")
SR_DELETE_MODEL_FOLDER     = os.path.join(OUTPUT_ROOT, "2full")

# ── Visualization output paths ───────────────────────────────────────────────
SR_VIS_OUT_TRAINVAL    = os.path.join(YB_ROOT, "out")               # trainVAL.py plt.savefig
SR_VIS_OUT_TG          = os.path.join(YB_ROOT, "out")               # tg.py plt.savefig
SR_VIS_OUT_TRAIN       = os.path.join(YB_UU, "Img")                 # train.py plt.savefig
SR_VIS_TEMP_DIR        = os.path.join(YB_ROOT, "temp")              # trainNoWave.py / trainVAL.py temporary png

# genvideo.py
SR_GENVIDEO_OUT        = os.path.join(YB_ROOT, "result/video/gt.mp4")

# tesg.py (top-level)
SR_TESG_IMAGE          = os.path.join(YB_EUR, "ctlow/712l.png")
SR_TESG_MODEL_DIR      = os.path.join(YB_ROOT, "model/ganmodel")
SR_TESG_RESULT_DIR     = os.path.join(YB_ROOT, "result/best/712/")
SR_TESG_FOLDER         = os.path.join(YB_ROOT, "result/best/1212")
SR_TESG_GT             = os.path.join(YB_EUR, "cttrue/1212.png")
SR_TESG_PSNR_PNG       = os.path.join(YB_ROOT, "picr/psnr1212bestl.png")

# ── Ui/ subdirectory paths ───────────────────────────────────────────────────
UI_MODEL_DIR    = os.path.join(YB_ROOT, "Ui/model")
UI_CTLOW_DIR    = os.path.join(YB_ROOT, "Ui/ctlow")
UI_CTTRUE_DIR   = os.path.join(YB_ROOT, "Ui/cttrue")
UI_RESULT_DIR   = os.path.join(YB_ROOT, "Ui/result")
UI_EXAMPLE_IMG  = os.path.join(YB_ROOT, "Ui/result/image12.png")
UI_CTTRUE_1212  = os.path.join(YB_ROOT, "Ui/cttrue/1212.png")

# Ui/tesg.py
UI_TESG_IMAGE      = os.path.join(YB_EUR, "ctlow/712l.png")
UI_TESG_MODEL_DIR  = os.path.join(YB_ROOT, "model/l1model")
UI_TESG_RESULT_DIR = os.path.join(YB_ROOT, "result/")
UI_TESG_FOLDER     = os.path.join(YB_ROOT, "result")
UI_TESG_GT         = os.path.join(YB_EUR, "cttrue/712.png")

# Ui/test.py (old environment /home/weiyb/UU)
UI_TEST_IMAGE      = os.path.join(WEIYB_ROOT, "ctlow/502.png")
UI_TEST_MODEL_DIR  = os.path.join(WEIYB_ROOT, "model")
UI_TEST_RESULT_DIR = os.path.join(WEIYB_ROOT, "result/")
UI_TEST_IMG1       = os.path.join(WEIYB_ROOT, "cttrue/502.png")
UI_TEST_IMG2       = os.path.join(WEIYB_ROOT, "result/502.png")

# Ui/resiez.py (old environment)
UI_RESIEZ_IN       = os.path.join(WEIYB_ROOT, "cttrue/502.png")
UI_RESIEZ_OUT      = os.path.join(WEIYB_ROOT, "ctlow/502.png")

# ── check/ subdirectory paths ────────────────────────────────────────────────
CHECK_OUTPUT_DIR       = os.path.join(YB_ROOT, "check/result")
CHECK_TRAIN_IMG_DIR    = os.path.join(DATA_ROOT, "check/train/images")
CHECK_TRAIN_LABEL_DIR  = os.path.join(DATA_ROOT, "check/train/labels")
CHECK_ANNOTATED_DIR    = os.path.join(DATA_ROOT, "check/train/annotated_images")

# ── Windows debug paths (only in __main__ blocks, not for production) ────────
DEBUG_INPUT_RESNET  = r"D:\datasetdata\total qing ct\1.png"   # Resnet152.py __main__
DEBUG_DEGRAD_IN     = r"D:\tryok\A002"                         # degradtion.py __main__
DEBUG_DEGRAD_OUT    = r"D:\tryok\outy"                         # degradtion.py __main__
