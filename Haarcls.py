import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 数据集路径
dataset_path = 'zibi_images'
# 图像统一大小
image_size = (128,128)  # 统一大小为64x64

# 用于存储图像和标签的列表
X = []
y = []

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
# 转换为NumPy数组
X = np.array(X)
y = np.array(y)

# 预处理图像并提取Haar特征
# 这里我们将每张图像转换为灰度，然后应用Haar特征提取
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 提取Haar特征函数
def extract_haar_features(images):
    haar_features = []
    for img in images:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:  # 如果检测到人脸
            # 选取检测到的第一个人脸区域作为特征
            x, y, w, h = faces[0]
            face_region = gray_image[y:y+h, x:x+w]
            face_region_resized = cv2.resize(face_region, (128,128))  # 调整大小以一致性
            haar_features.append(face_region_resized.flatten())  # 展平为一维数组
        else:
            haar_features.append(np.zeros((128*128,)))  # 未检测到人脸，填充零
    return np.array(haar_features)

# 提取Haar特征
X_haar = extract_haar_features(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_haar, y, test_size=0.2, random_state=42)

# 使用Adaboost训练分类器
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# 基学习器
base_estimator = DecisionTreeClassifier(max_depth=1)  # 单层决策树
clf_haar = AdaBoostClassifier(estimator=base_estimator, n_estimators=100, random_state=42)

# 训练模型
clf_haar.fit(X_train, y_train)
#保存模型
import joblib
joblib.dump(clf_haar, 'haar_model.pkl')


# 在测试集上进行预测
y_pred = clf_haar.predict(X_test)

# 计算准确率
accuracy_haar = accuracy_score(y_test, y_pred)
print(f"Haar特征模型的准确率: {accuracy_haar:.2f}")
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