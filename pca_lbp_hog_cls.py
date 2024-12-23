#PCA+LBP+HOG+RF/SVM
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from skimage.feature import hog, local_binary_pattern

# 数据集路径
dataset_path = 'zibi_images'

# 用于存储图像和标签的列表
X = []
y = []

# 图像统一大小
image_size = (224,224)  # 统一大小为64x64


# 加载数据集
def load_dataset(dataset_path):
    non_autistic_path = os.path.join(dataset_path, 'Non_Autistic')
    autistic_path = os.path.join(dataset_path, 'Autistic')

    # 处理非自闭症儿童的图片
    for image_name in os.listdir(non_autistic_path):
        image_path = os.path.join(non_autistic_path, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image_resized = cv2.resize(image, image_size)  # 调整图像大小
            X.append(image_resized)
            y.append(0)  # 标签为0表示非自闭症

    # 处理自闭症儿童的图片
    for image_name in os.listdir(autistic_path):
        image_path = os.path.join(autistic_path, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image_resized = cv2.resize(image, image_size)  # 调整图像大小
            X.append(image_resized)
            y.append(1)  # 标签为1表示自闭症


load_dataset(dataset_path)

# 将列表转换为NumPy数组
X = np.array(X)
y = np.array(y)


# 特征提取函数结合LBP和HOG特征
def extract_features(image):
    radius = 3
    n_points = 8 * radius
    lbp_features = local_binary_pattern(image, n_points, radius, method='uniform')

    hog_features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                          block_norm='L2-Hys', visualize=True, transform_sqrt=True)

    return np.hstack([lbp_features.ravel(), hog_features.ravel()])


# 预处理图像并提取特征
processed_images = []
for image in X:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    processed_image = extract_features(gray_image)
    processed_images.append(processed_image)

X_processed = np.array(processed_images)

# 应用PCA进行降维
pca = PCA(n_components=100)  # 根据实际情况调整组件数
X_pca = pca.fit_transform(X_processed)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# # 训练随机森林分类器
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)
#
# # 在测试集上预测
# y_pred = clf.predict(X_test)

# # 计算准确率
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: {:.2f}".format(accuracy))
# 计算SVM的准确率
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# SVC分类器
clf = SVC()
clf.fit(X_train, y_train)
#保存模型
from joblib import dump, load
dump(clf, 'model.joblib')

# 在测试集上预测
y_pred = clf.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))
#计算 Sensitivity 和 Specificity
from sklearn.metrics import confusion_matrix
# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
#保存混淆矩图
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig('confusion_matrix.png')

# 计算Sensitivity和Specificity
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print(f'Sensitivity: {sensitivity:.2f}')
print(f'Specificity: {specificity:.2f}')
#计算精确率、召回率和F1
from sklearn.metrics import precision_score, recall_score, f1_score
# 计算精确率
precision = precision_score(y_test, y_pred)
# 计算召回率
recall = recall_score(y_test, y_pred)
# 计算F1
f1 = f1_score(y_test, y_pred)
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1: {f1:.2f}')
#画ROC曲线
from sklearn.metrics import roc_curve, roc_auc_score
# 计算ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred)
# 计算AUC
auc = roc_auc_score(y_test, y_pred)
# 画ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('roc.png')
#输出auc
print(f'AUC: {auc:.2f}')



