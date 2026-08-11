import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)
#提取MNIST数据集，作为训练集和测试集，训练集训练，测试集不训练；均需要下载；都需要转化张量；

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
##shuffle 用于表示是否打乱，训练集需要打乱，测试集不需要
#batch_size 用于表示每一次训练的图片数量，即每一轮的训练量

print(f"训练集大小：{len(train_dataset)}张图片")
print(f"测试集大小：{len(test_dataset)}张图片")

class SimpleCNN(nn.Module):##继承自nn.Module
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        ##卷积层1，2, 3

        self.pool = nn.MaxPool2d(2,2)##池化层，每次池化2*2区域，步长为2
        self.fc1 = nn.Linear(64*3*3, 128)
        #全联接层：其实此处正是转接矩阵，具有欧米伽和b两个变量
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64*3*3)
        #把矩阵变成一条长列
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN()
print(model)

#定义损失函数与优化器：
criterion = nn.CrossEntropyLoss()
#交叉熵损失，适合分类问题
optimizer = optim.Adam(model.parameters(), lr=0.001)
#使用adam优化器，学习率0.001



##开始循环训练：
num_epochs = 10

print("\n开始训练")
print("=" * 50)

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        #清零梯度:防止梯度累加
        optimizer.zero_grad()

        #前向传播：图片进模型，得到预测结果
        outputs = model(images)

        #计算损失：预测与真实标签的差距
        loss = criterion(outputs, labels)#交叉熵损失函数

        #反向传播：计算梯度
        loss.backward()#交叉熵损失函数的自带反向传播

        #更新参数：根据梯度调整权重
        optimizer.step()

        #统计信息
        running_loss += loss.item()#总损失
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)#平均损失
    epoch_acc = correct / total * 100 #百分数
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

print("训练完成")
print("=" * 50)

