import torch
import torch.nn as nn
import torch.optim as op
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
)
test_dataset  = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"训练集：{len(train_loader)}")
print(f"测试集：{len(test_dataset)}")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)  ##池化层，每次池化2*2区域，步长为2
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        # 全联接层：其实此处正是转接矩阵，具有欧米伽和b两个变量
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = self.pool(torch.tanh(self.conv3(x)))
        x = x.view(-1, 64*3*3)
        #把矩阵变成一条长列
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN()
print(model)

#定义损失函数与优化器：
criterion = nn.CrossEntropyLoss()
#交叉熵损失，适合分类问题
optimizer = op.Adam(model.parameters(), lr=0.001)
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

print("开始测试！")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"测试集准确率：{100 * correct / total:.2f}%")