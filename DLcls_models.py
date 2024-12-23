#SOME MODELS FOR FACE CLASSIFICATION
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import metrics
import torchvision.models as models
import timm

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = os.listdir(root_dir)
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = 0 if 'Non' in self.root_dir else 1
        return img, label


#定义数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset1 = FaceDataset('zibi/Non_Autistic', transform=transform)
dataset2 = FaceDataset('zibi/Autistic', transform=transform)

# 合并数据集
dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
#划分数据集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

#加载数据集
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

#CHANGE THE MODEL HERE
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #google net
        self.model = models.googlenet(pretrained=False)
        self.model.fc = nn.Linear(1024, 2)
        #resnet

    def forward(self, x):
        return self.model(x)

def train(model, train_loader, test_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(f'epoch {epoch}, loss {loss.item()}')

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'epoch {epoch}, accuracy {correct / total}')


#测试
def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        #ACC：
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'final accuracy {correct / total}')
        #AUC:画ROC曲线
        all_labels = []
        all_preds = []
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        fpr, tpr, _ = metrics.roc_curve(all_labels, all_preds)
        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.savefig('roc.png')
        print(f'AUC {metrics.auc(fpr, tpr)}')
        #Sensitivity, Specificity
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        TP = np.sum((all_labels == 1) & (all_preds >= 0.5))
        TN = np.sum((all_labels == 0) & (all_preds < 0.5))
        FP = np.sum((all_labels == 0) & (all_preds >= 0.5))
        FN = np.sum((all_labels == 1) & (all_preds < 0.5))
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        print(f'Sensitivity {sensitivity}, Specificity {specificity}')
        #精确率，召回率，F1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        print(f'Precision {precision}, Recall {recall}, F1 {f1}')
        #混淆矩阵，画图的方式保存
        confusion_matrix = np.zeros((2, 2))
        confusion_matrix[0, 0] = TN
        confusion_matrix[0, 1] = FP
        confusion_matrix[1, 0] = FN
        confusion_matrix[1, 1] = TP
        plt.imshow(confusion_matrix)
        plt.savefig('confusion_matrix.png')

#训练
model = Net()
train(model, train_loader, test_loader, num_epochs=30)
#测试
test(model, test_loader)




