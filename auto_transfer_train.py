# 引入模块
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
''' 模型迁移 -> 导入库 ''' 
import torch_npu
# 使用 apex 的 AMP 模块
# from apex import amp 
# 推荐使用框架内置的 AMP 模块
# from torch.cuda import amp # 会被 transfer_to_npu 自动替换为 npu 的 amp
from torch_npu.npu import amp
# 使能自动迁移，即将 gpu 的 api 更改为 npu 的 api
from torch_npu.contrib import transfer_to_npu    

# 初始化运行device
device = torch.device('cuda:0')   

# 定义模型网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            # 卷积层
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            # 池化层
            nn.MaxPool2d(kernel_size=2),
            # 卷积层
            nn.Conv2d(16, 32, 3, 1, 1),
            # 池化层
            nn.MaxPool2d(2),
            # 将多维输入一维化
            nn.Flatten(),
            nn.Linear(32*7*7, 16),
            # 激活函数
            nn.ReLU(),
            nn.Linear(16, 10)
        )
    def forward(self, x):
        return self.net(x)

# 下载数据集
train_data = torchvision.datasets.MNIST(
    root='mnist',
    download=True,
    train=True,
    transform=torchvision.transforms.ToTensor()
)

# 定义训练相关参数
batch_size = 64   
model = CNN().to(device) # 定义模型
train_dataloader = DataLoader(train_data, batch_size=batch_size) # 定义DataLoader   
loss_func = nn.CrossEntropyLoss().to(device)    # 定义损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)    # 定义优化器
''' 模型迁移 -> 开启混合精度 ''' 
scaler = amp.GradScaler() # 在模型、优化器定义之后，定义 GradScaler
epochs = 10 

# 设置循环
for epoch in range(epochs):
    print(f"start training epoch: {epoch}")
    for imgs, labels in train_dataloader:
        start_time = time.time()   
        imgs = imgs.to(device)    
        labels = labels.to(device)    
        ''' 模型迁移 -> 混合精度计算 ''' 
        with amp.autocast(): # 工作机制是拷贝 float32 模型转为 float16 进行前向计算
            outputs = model(imgs) # 前向计算
            loss = loss_func(outputs, labels) # 损失函数计算
        optimizer.zero_grad()
        # loss.backward() # 损失函数反向计算
        # optimizer.step() # 更新优化器
        ''' 模型迁移 -> 进行反向传播前后的 loss 缩放、参数更新 '''
        scaler.scale(loss).backward()    # loss 缩放并反向转播
        scaler.step(optimizer)    # 更新参数（自动 unscaling）
        scaler.update()    # 基于动态 Loss Scale 更新 loss_scaling 系数

# 定义保存模型
torch.save({'epoch': 10,
            'arch': CNN,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }, 
            'checkpoint.pth.tar')
