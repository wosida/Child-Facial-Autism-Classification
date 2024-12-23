import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import torch.optim as optim
from sklearn.model_selection import KFold
#from PH_Trans_mamba import PHMBA
#from SPFFM import SPFFM
class GoogLeNet(nn.Module):
    # 传入的参数中aux_logits=True表示训练过程用到辅助分类器，aux_logits=False表示验证过程不用辅助分类器
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        #self.phmba = PHMBA()

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        #self.spffm = SPFFM(img_size=(14,14), patch_size=(2, 2), in_channels=512, dim=128, out_channels=512, num_heads=8, cnn_drop=0.1, vit_drop=0.1, down_sampling=False)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        # N x 3 x 224 x 224
        #x = self.phmba(x)
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        #x=self.spffm(x)
        if self.training and self.aux_logits:  # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:  # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7
        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:  # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.constant_(m.bias, 0)


# Inception结构
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)  # 按 channel 对四个分支拼接


# 辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


# 基础卷积层（卷积+ReLU）
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 过滤掉 .ipynb_checkpoints 和非图片文件
        self.imgs = [f for f in os.listdir(root_dir) if
                     os.path.isfile(os.path.join(root_dir, f)) and not f.startswith('.ipynb_checkpoints')]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = 0 if 'Non' in self.root_dir else 1
        return img, label

def train(model, train_loader, test_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            logits, aux_logits2, aux_logits1 = model(inputs)
            loss0 = criterion(logits, labels.to(device))
            loss1 = criterion(aux_logits1, labels.to(device))
            loss2 = criterion(aux_logits2, labels.to(device))
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
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
    result={}
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
        #返回所有指标
        result['accuracy']=correct / total
        result['AUC']=metrics.auc(fpr, tpr)
        result['Sensitivity']=sensitivity
        result['Specificity']=specificity
        result['Precision']=precision
        result['Recall']=recall
        result['F1']=f1
        return result

if __name__ == '__main__':


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset1 = FaceDataset('zibi/Non_Autistic', transform=transform)
    dataset2 = FaceDataset('zibi/Autistic', transform=transform)


    dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
    results=[]
    #10折交叉验证
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(dataset):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        test_dataset = torch.utils.data.Subset(dataset, test_index)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
        model = GoogLeNet(num_classes=2, aux_logits=True, init_weights=True)
        train(model, train_loader, test_loader, num_epochs=10)
        result=test(model, test_loader)
        results.append(result)


    #计算平均值
    accuracy=0
    AUC=0
    Sensitivity=0
    Specificity=0
    Precision=0
    Recall=0
    F1=0
    for result in results:
        accuracy+=result['accuracy']
        AUC+=result['AUC']
        Sensitivity+=result['Sensitivity']
        Specificity+=result['Specificity']
        Precision+=result['Precision']
        Recall+=result['Recall']
        F1+=result['F1']
    print('accuracy:',accuracy/10)
    print('AUC:',AUC/10)
    print('Sensitivity:',Sensitivity/10)
    print('Specificity:',Specificity/10)
    print('Precision:',Precision/10)
    print('Recall:',Recall/10)
    print('F1:',F1/10)



    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    #
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = GoogLeNet(num_classes=2, aux_logits=True, init_weights=True)
    # model.to(device)
    # train(model, train_loader, test_loader, num_epochs=10)
    #
    #
    # test(model, test_loader)





