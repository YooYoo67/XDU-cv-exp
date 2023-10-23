import numpy as np
import cv2
import math


def sum_(img, kernel):
    res = (img * kernel).sum()
    if (res < 0):
        res = 0
    elif (res > 255):
        res = 255
    return res


def cross_correlation_2d(img, kernel):  # 互相关函数 参数为已经扩展的图片和权重核
    img_array = np.array(img)
    img_array_row = img_array.shape[0]  # 图像的行
    img_array_column = img_array.shape[1]  # 图像的列
    img_array_vim = img_array.shape[2]  # 图像的维度

    kernel_row = kernel.shape[0]  # 核的行
    kernel_column = kernel.shape[1]  # 核的列

    lift_and_right = np.zeros((img_array_row, kernel_column // 2), int)
    up_and_down = np.zeros((kernel_row // 2, img_array_column + lift_and_right.shape[1] * 2), int)

    conv = np.zeros((img_array_row, img_array_column, img_array_vim))

    for i in range(3):
        temp_img_array = np.hstack([lift_and_right, np.hstack([img_array[:, :, i], lift_and_right])])

        new_img_array = np.vstack([up_and_down, np.vstack([temp_img_array, up_and_down])])

        for j in range(img_array_row):
            for k in range(img_array_column):
                conv[j][k][i] = sum_(new_img_array[j:j + kernel_row, k:k + kernel_column], kernel)

    return conv


def convolve_2d(img, kernel):  # 卷积函数 权重核左右和上下翻转
    kernel_flipped = np.flipud(np.fliplr(kernel))
    return cross_correlation_2d(img, kernel_flipped)


def gaussian_blur_kernel_2d(sigma, height, width):  # 高斯核生成函数
    center_row = height / 2
    center_column = width / 2
    s = 2 * (sigma ** 2)
    gaussian_kernel = np.zeros((height, width), dtype='double')
    for i in range(height):
        for j in range(width):
            x = i - center_row
            y = j - center_column
            gaussian_kernel[i][j] = (1.0 / (np.pi * s)) * np.exp(-float(x ** 2 + y ** 2) / s)

    return gaussian_kernel


def low_pass(img, sigma, size):  # 低通函数 使用高斯核与图像卷积得到低通图像
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size[0], size[1]))


def high_pass(img, sigma, size):  # 高通函数 原图减去低通图像
    img_array = np.array(img)
    return img_array - low_pass(img, sigma, size)


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


image1 = cv2.imread("dog.jpg")
image2 = cv2.imread("cat.jpg")
ratio = 0.55
im1_hy = low_pass(image1, 5, (19, 19))
im2_hy = high_pass(image2, 6, (15, 15))

im_add = create_hybrid_image(image1, image2, 5, (19, 19), 'low', 6, (15, 15), 'high', ratio)

cv2.imwrite("left.png", im1_hy)
cv2.imwrite("right.png", im2_hy)
cv2.imwrite("hybrid.png", im_add)
cv2.waitKey(0)
cv2.destroyAllWindows()
