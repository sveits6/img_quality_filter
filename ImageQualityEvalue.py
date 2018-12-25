# -*- coding:utf-8 -*-
import cv2
import time
import numpy as np
from scipy.ndimage import filters

# DCT: sharpness
def get_dct(img):
    dct = cv2.dct(img)
    s = img.shape
    idct = np.zeros(s, dtype='float32')
    idct[:s[0] * 3 / 4, :s[1] * 3 / 4] = dct[:s[0] * 3 / 4, :s[1] * 3 / 4]
    idct = cv2.idct(idct)
    return np.square(idct - img).mean()


# Histogram: symmetry
def his_sym(img):
    his1 = cv2.calcHist(img[:, :30], [0], None, [16], [0., 255.])
    his2 = cv2.calcHist(img[:, 30:], [0], None, [16], [0., 255.])
    return np.square(his1 - his2).mean()


# Histogram: contrst
def his_con(img):
    his = cv2.calcHist(img, [0], None, [16], [0., 255.])
    return np.square(his - his.mean()).mean()


# Sobel: sharpness
def get_sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    z = cv2.Sobel(img, cv2.CV_16S, 1, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    absZ = cv2.convertScaleAbs(z)
    dst1 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    dst = cv2.addWeighted(absZ, 0.5, dst1, 0.5, 0)
    return dst.mean()

# Nuclear norm
def nu_norm(img):
    return np.linalg.norm(img, ord='nuc')

def luminance(img):
    return img.mean()
# Focus Score
def focus(img):
    kernel_xx = [[1, -2, 1]]
    imx = np.zeros(img.shape)
    filters.convolve(img, kernel_xx, imx)
    kernel_yy = [[1], [-2], [1]]
    imy = np.zeros(img.shape)
    filters.convolve(img, kernel_yy, imy)
    sumvalue =  np.abs(imx).sum() + np.abs(imy).sum()
    # imgh = img.shape[0]
    # imgw = img.shape[1]
    # everageValue = np.abs(imx).mean() + np.abs(imy).mean()
    return sumvalue
def getLaplacian(img):
    clearify = cv2.Laplacian(img, cv2.CV_64F).var()
    return clearify
def image_quality_evalue(image, lum=0, contrast = 0, soble = 0, lap =0 ):
    # stat = time.time()
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    lum = luminance(image_gray)
    image2 = cv2.resize(image_gray, (64, 64))
    foucsv = focus(image2)
    foucsv = foucsv * 1.0 / (4096)
    # contrast = his_con(image)
    # soble = get_sobel(image)
    # lap = getLaplacian(image)
    # endt = time.time()
    # print('get img quality:{}'.format(endt - stat))
    return lum,foucsv,contrast,soble,lap

def main():
    video_path = '/data_b/data_alpen/face_rec/test.mp4'
    frame_dir = '/data_b/data_alpen/face_rec/test8'
    get_frame_opencv(video_path, frame_dir)

if __name__ == "__main__":
    main()