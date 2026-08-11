import torch
import torch.nn as nn
import torch.optim as op
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    #数据打包：
    transforms.ToTensor(),
    #转化为张量：从0-1
    transforms.Normalize((0.1307,), (0.3081,))
    #归一化：把数据转变成均值≈0，方差≈1，加速神经网络收敛
])

train_dataset = datasets.MNIST(
    root = './data',
    train = True,
    transform = transform,
    download = True
)

test_dataset = datasets.MNIST(
    root = './data',
    train = False,
    transform = transform,
    download = True
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


print(f"训练集大小：{len(train_dataset)}")
print(f"测试集大小：{len(test_dataset)}")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        #继承父类的init函数
        self.conv = nn.Conv2d(1,32,kernel_size = 3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(-1,32*14*14)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(torch.relu(x))
        x = self.fc3(x)
        return x

model = SimpleCNN()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = op.Adam(model.parameters(), lr = 0.001)

num_epochs = 10
print("\n开始训练")
print("=" * 50)

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        #torch.max是PyTorch中的最大值函数，有两种用法：
        #1.直接取input，取整个张量里的全局最大值，值是一个标量
        #2.取（input，dim）表示输入和维度，在指定维度（dim）上取最大值，并告诉你最大值是多少、在哪个位置
        #dim=1是横向比较（排），dim=0纵向比较（列）
        #其返回值是一个张量，（最大值张量，索引张量）
        #此处的predicted取的是后面一个值，其实是一个tensor，其中它内部包含了单个batch（64）个推测值
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #拿推测值和标准值比较

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total * 100
    print(f"Epoch[{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

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