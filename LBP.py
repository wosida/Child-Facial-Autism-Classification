import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# 1. 读取图像并灰度化
image_path = 'image.jpg'  # 替换为你的图片路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 2. LBP 参数
radius = 1  # 邻域半径
n_points = 8 * radius  # 邻域像素点数

# 3. 计算 LBP 特征
lbp = local_binary_pattern(image, n_points, radius, method='uniform')

# 4. 显示原始图像和 LBP 特征图像
plt.figure(figsize=(10, 5))

# 原始图像
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# LBP 特征图像
plt.subplot(1, 2, 2)
plt.imshow(lbp, cmap='gray')
plt.title('LBP Image')
plt.axis('off')

plt.show()
