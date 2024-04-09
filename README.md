# Atlas 800(9000) PyTorch 模型训练
若用户使用Atlas 训练系列产品，由于其架构特性限制，用户在**训练时需要开启混合精度（AMP）**，可以提升模型的性能。若用户使用Atlas A2 训练系列产品，则可以选择不开启混合精度（AMP）。
# 自动迁移
（1）导入必要模块
```
''' 模型迁移 -> 导入库 ''' 
import torch_npu
# 使用 apex 的 AMP 模块
# from apex import amp 
# 推荐使用框架内置的 AMP 模块
# from torch.cuda import amp # 会被 transfer_to_npu 自动替换为 npu 的 amp
from torch_npu.npu import amp
# 使能自动迁移，即将 gpu 的 api 更改为 npu 的 api
from torch_npu.contrib import transfer_to_npu 
```
（2）开启混合精度
```
''' 模型迁移 -> 开启混合精度 ''' 
scaler = amp.GradScaler() # 在模型、优化器定义之后，定义 GradScaler
```
（3）混合精度前向计算
```
''' 模型迁移 -> 混合精度计算 '''
with amp.autocast(): # 工作机制是拷贝 float32 模型转为 float16 进行前向计算
     outputs = model(imgs) # 前向计算
     loss = loss_func(outputs, labels) # 损失函数计算
```
（4）混合精度梯度缩放反向传播
```
''' 模型迁移 -> 进行反向传播前后的 loss 缩放、参数更新 '''
scaler.scale(loss).backward()    # loss 缩放并反向转播
scaler.step(optimizer)    # 更新参数（自动 unscaling）
scaler.update()    # 基于动态 Loss Scale 更新 loss_scaling 系数
```


参考：
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/ptmoddevg/ptmigr/AImpug_0002.html