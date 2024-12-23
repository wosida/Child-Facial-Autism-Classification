import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces

# 1. 加载Olivetti人脸数据集
faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)
images = faces_data.images  # 原始图像 (n_samples, height, width)
data = faces_data.data  # 展平后的图像 (n_samples, n_features)

# 2. 使用PCA提取特征
n_components = 100  # 提取100个主成分
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(data)
pca_faces = pca.transform(data)  # 转换为PCA特征

# 3. 可视化前几个主成分（特征脸）
eigenfaces = pca.components_.reshape((n_components, images.shape[1], images.shape[2]))

plt.figure(figsize=(15, 6))
for i in range(10):  # 显示前10个主成分
    plt.subplot(2, 5, i + 1)
    plt.imshow(eigenfaces[i], cmap='gray')
    plt.title(f'PC {i + 1}')
    plt.xticks([])
    plt.yticks([])
plt.suptitle("Top 10 PCA Components (Eigenfaces)")
plt.show()

# 4. 重建人脸图像
reconstructed_faces = pca.inverse_transform(pca_faces)

plt.figure(figsize=(10, 5))
for i in range(5):
    # 显示原始图像
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])

    # 显示重建图像
    plt.subplot(2, 5, i + 6)
    plt.imshow(reconstructed_faces[i].reshape(64, 64), cmap='gray')
    plt.title('Reconstructed')
    plt.xticks([])
    plt.yticks([])
plt.show()
