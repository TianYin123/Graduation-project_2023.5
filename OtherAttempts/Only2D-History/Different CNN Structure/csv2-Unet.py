
import numpy as np
import torch
import os
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
random.seed(3407)
rmb_label = {"ASD": 0, "TD": 1}      # 设置标签

class ABIDEtxtDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        ABIDE_db的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"ASD": 0, "TD": 1}
        self.data_info = self.get_txt_info(data_dir)  # data_info存储所有txt路径和标签，在DataLoader中通过index读取样本
        self.transform = transform
        print('Number of samples:', len(self.data_info))
        print('Sample info:', self.data_info[0])
        
    def __getitem__(self, index):
        path_txt, label = self.data_info[index]
        txt_data = np.loadtxt(path_txt, delimiter=',')
        #转换为1channel
        # cor = torch.from_numpy(txt_data).float().unsqueeze(0)
        #转换为3channel
        cor = torch.from_numpy(txt_data).float().unsqueeze(0).repeat(3, 1, 1)
        
        if self.transform is not None:
            txt = 1   # 在这里做transform，转为tensor等等

        return cor, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_txt_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                txt_names = os.listdir(os.path.join(root, sub_dir))
                txt_names = list(filter(lambda x: x.endswith('.csv'), txt_names))

                # 遍历txt
                for i in range(len(txt_names)):
                    txt_name = txt_names[i]
                    path_txt = os.path.join(root, sub_dir, txt_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_txt, int(label)))

        return data_info


# # Add your DoubleConv and Down classes here.
# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             DoubleConv(in_channels, out_channels),
#             nn.MaxPool2d(2)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)
    
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             Down(1, 32),
#             Down(32, 64),
#             Down(64, 128),
#             Down(128, 256)
#         )
#         self.fc1 = nn.Linear(256*7*7,256)
#         self.fc2 = nn.Linear(256, 2)
#     def forward(self, x):
#         x = self.encoder(x)
#         x = x.view(-1, 256*7*7)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
        # return x

# 构建模型
class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.features = models.vgg11(pretrained=False).features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 初始化模型和损失函数
model = VGG11()


train_losses = []
train_accs = []
valid_losses = []
valid_accs = []
split_dir = os.path.join('easy/not modify/csv_split')  
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")
test_dir  = os.path.join(split_dir, "test") 
device = torch.device('cpu')
num_workers = 1

train_transform1 = transforms.Compose([
    #1.上下翻转
    transforms.RandomVerticalFlip(p = 1),
    #2.左右翻转
    transforms.RandomHorizontalFlip(p = 1),
    transforms.ToTensor(),
    
])
train_transform2 = transforms.Compose([
    #1.上下翻转
    transforms.RandomHorizontalFlip(p = 1),
    #2.左右翻转
    transforms.RandomVerticalFlip(p = 1),
    transforms.ToTensor(),
])

train_transform3 = transforms.Compose([
    transforms.RandomCrop(116, padding=116, padding_mode='symmetric'),
    transforms.ToTensor(),
])


# 定义超参数
input_size = [116,116]
batch_size = 128
learning_rate = 0.001
num_epochs = 50
hidden_size = 256
num_layers = 8
num_classes = 2
log_interval = 8
val_interval = 8

def collate_fn(batch):
    data = [d[0] for d in batch]
    label = [d[1] for d in batch]
    # 为数据添加一个额外的维度，以便将其转换为形状为 [batch_size, 1, height, width] 的张量
    data = [d.unsqueeze(0) for d in data]
    # 将多个数据合并为一个batch
    data = torch.stack(data, dim=0)
    label = torch.tensor(label)
    data = data[:len(label)]
    return data, label

model = VGG11()
# model = SimpleCNN()
# model = SimpleCNN_SE()
# model = CustomResNet18()
# model.initialize()

# dataset0 = ABIDEtxtDataset(train_dir, transform=None)
# dataset1 = ABIDEtxtDataset(train_dir, transform=train_transform1)
# dataset2 = ABIDEtxtDataset(train_dir, transform=train_transform2)
# dataset3 = ABIDEtxtDataset(train_dir, transform=train_transform3)

# # 重复原始数据集
# train_dataset = ConcatDataset([dataset0,dataset1, dataset2,dataset3])

# 实例化数据集和数据加载器
train_dataset = ABIDEtxtDataset(train_dir, transform=None)
valid_dataset = ABIDEtxtDataset(valid_dir, transform=None)
#CNN 图像模型dataloader
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn = collate_fn,num_workers = num_workers)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,collate_fn = collate_fn,num_workers = num_workers)
#非CNN 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# 实例化模型、损失函数和优化器
# model = SqueezeNet2D(2)
# model = SimpleCNN()
# model.initialize
# model = RNNModel(input_size,hidden_size,num_layers,num_classes)
# model = SAE(input_size,hidden_size)


model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)     # 设置学习率下降策略

print('训练')
# ============================ step 5/5 训练 ============================
# 训练和验证
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    train_loss = running_loss / len(train_dataset)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # 验证阶段
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        total = 0
        correct = 0
        for data, label in valid_loader:
            data, label = data.to(device), label.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, label)
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            # pred = outputs.argmax(dim=1)
            # print(f"Sample: predicted={pred.item()}, target={label.item()},matched?={label.item() == pred.item()}")

    valid_loss = running_loss / len(valid_dataset)
    valid_acc = correct / total
    valid_losses.append(valid_loss)
    valid_accs.append(valid_acc)

    # 打印训练过程中的指标
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.4f}")


# 绘制训练和验证指标的图表
fig, axs = plt.subplots(2, 1, figsize=(10, 12))
axs[0].plot(train_accs, label='Train Accuracy')
axs[0].plot(valid_accs, label='Validation Accuracy')
axs[0].legend()
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Value')
axs[0].set_title('Training and Validation Metrics')

axs[1].plot(train_losses, label='Train Loss')
axs[1].plot(valid_losses, label='Validation Loss')
axs[1].legend()
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Value')
axs[1].set_title('Training and Validation Metrics')

plt.show()