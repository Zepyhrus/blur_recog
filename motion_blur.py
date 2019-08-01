import math
import numpy as np
import numpy.random as npr
import cv2


# 生成卷积核和锚点
def genaratePsf(length, angle):
    half = length / 2
    EPS = np.finfo(float).eps
    alpha = (angle - math.floor(angle / 180) * 180) / 180 * math.pi
    cosalpha = math.cos(alpha)
    sinalpha = math.sin(alpha)
    if cosalpha < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:
        xsign = 1
    psfwdt = 1
    # 模糊核大小
    sx = int(math.fabs(length * cosalpha + psfwdt * xsign - length * EPS))
    sy = int(math.fabs(length * sinalpha + psfwdt - length * EPS))
    psf1 = np.zeros((sy, sx))

    # psf1是左上角的权值较大，越往右下角权值越小的核。
    # 这时运动像是从右下角到左上角移动
    for i in range(0, sy):
        for j in range(0, sx):
            psf1[i][j] = i * math.fabs(cosalpha) - j * sinalpha
            rad = math.sqrt(i * i + j * j)
            if rad >= half and math.fabs(psf1[i][j]) <= psfwdt:
                temp = half - math.fabs((j + psf1[i][j] * sinalpha) / cosalpha)
                psf1[i][j] = math.sqrt(psf1[i][j] * psf1[i][j] + temp * temp)
            psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j])
            if psf1[i][j] < 0:
                psf1[i][j] = 0
    # 运动方向是往左上运动，锚点在（0，0）
    anchor = (0, 0)
    # 运动方向是往右上角移动，锚点一个在右上角
    # 同时，左右翻转核函数，使得越靠近锚点，权值越大
    if angle < 90 and angle > 0:
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, 0)
    elif angle > -90 and angle < 0:  # 同理：往右下角移动
        psf1 = np.flipud(psf1)
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, psf1.shape[0] - 1)
    elif angle < -90:  # 同理：往左下角移动
        psf1 = np.flipud(psf1)
        anchor = (0, psf1.shape[0] - 1)
    psf1 = psf1 / psf1.sum()
    return psf1, anchor


def MotionBlur(image, blur_length, blur_angle):
    kernel, anchor = genaratePsf(blur_length, blur_angle)
    motion_blur = cv2.filter2D(image, -1, kernel, anchor=anchor)
    return motion_blur


def motion_blur(image, degree=12, angle=45):
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(
        motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def random_blur(img, level):
  kernel_size = 2 * level + 1
  blur_func = npr.choice(["gauss", "mean", "median", "motion"])

  if blur_func == "gauss":
    img_blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
  elif blur_func == "mean":
    img_blurred = cv2.blur(img, (kernel_size, kernel_size))
  elif blur_func == "median":
    img_blurred = cv2.medianBlur(img, kernel_size)
  elif blur_func == "motion":
    motion_len = npr.randint(2*level+1, 2*level+3)
    motion_ang = npr.randint(360)

    img_blurred = motion_blur(img, motion_len, motion_ang)

  return img_blurred


if __name__ == '__main__':
    test_file = "blur_cam_test/0/1501855183567___c_77_y_-2__p_-21__r_-9_aligned.png"
    image = cv2.imread(test_file)
    if image is not None:
        motion_blur = random_blur(image, 9)
        cv2.imshow("motion_blur", motion_blur)
        cv2.waitKey(0)
    else:
        print("Image does not exist!")
