
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2
best_psnr=0
best_ssim=0
def calculate(img1,img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    p = compare_psnr(img1, img2)
    s = compare_ssim(img1, img2, win_size=3, multichannel=True)
    return float(p),float(s)