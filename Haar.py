import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度
image = cv2.imread('1.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 创建积分图
integral_image = cv2.integral(gray_image)
# 图像统一大小
image_size = (224,224)  # 统一大小为64x64
# 定义Haar特征的矩形区域
def haar_feature(integral_img, x, y, w, h):
    # 这里以一个简单的边缘特征为例
    A = integral_img[y, x]
    B = integral_img[y, x + w]
    C = integral_img[y + h, x]
    D = integral_img[y + h, x + w]
    feature_value = (B + D) - (A + C)
    return feature_value

# 可视化Haar特征
def draw_haar_feature(image, x, y, w, h):
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# 示例位置和大小
x, y, w, h = 50, 50, 100, 50
feature_value = haar_feature(integral_image, x, y, w, h)

# 绘制矩形
draw_haar_feature(image.copy(), x, y, w, h)

# 显示结果
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f'Haar Feature Value: {feature_value}')
plt.axis('off')
plt.show()
