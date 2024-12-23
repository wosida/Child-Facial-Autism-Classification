import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# 1. 读取图像并转为灰度图
image_path = 'image.jpg'  # 替换为你的图片路径
image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 计算HOG特征和HOG图像
hog_features, hog_image = hog(image_gray,
                              orientations=9,
                              pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2),
                              block_norm='L2-Hys',
                              visualize=True,
                              feature_vector=True)

# 3. 可视化HOG图像
# 使用 skimage 提供的 exposure.rescale_intensity 将 HOG 图像进行增强
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# 4. 显示原始图像和HOG可视化图像
plt.figure(figsize=(10, 5))

# 显示原始图像
plt.subplot(1, 2, 1)
plt.imshow(image_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 显示HOG特征图像
plt.subplot(1, 2, 2)
plt.imshow(hog_image_rescaled, cmap='gray')
plt.title('HOG Image')
plt.axis('off')

plt.show()
