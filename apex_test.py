import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision

import torch_npu
from torch_npu.contrib import transfer_to_npu

''' 模型迁移 -> apex 混合精度计算 '''
import apex
    

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
batch_size = 128   
model = CNN().to(device) 
train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=4) 
loss_func = nn.CrossEntropyLoss().to(device)    
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) 
''' 模型迁移 -> apex 混合精度计算 '''
model, optimizer = apex.amp.initialize(model, optimizer, opt_level='O1', loss_scale=32.0)

epochs = 20 
for epoch in range(epochs):
    print(f"start training epoch: {epoch}")
    for imgs, labels in train_dataloader:
        start_time = time.time()   
        imgs = imgs.to(device)    
        labels = labels.to(device)             
        outputs = model(imgs) 
        loss = loss_func(outputs, labels)      
        optimizer.zero_grad()
        ''' 模型迁移 -> apex 混合精度计算 '''
        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:     
            scaled_loss.backward() 
        optimizer.step() 


# 定义保存模型
torch.save({'epoch': 10,
            'arch': CNN,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }, 
            'checkpoint.pth.tar')
