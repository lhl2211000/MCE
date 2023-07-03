import torch
import torch.nn as nn
from torch.nn import Parameter
"""
本实验为了测试torch.nn.paraneter.Parameter:
在定义神经网络时设置self.para=Parameter(torch.Tensor(...)),
这个Parameter将会出现在网络的parameters()迭代器中
"""
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.l1=nn.Linear(3,3,bias=False)#bias设置为True是为了简化网络模型的参数列表
        self.l2 = nn.Linear(3, 3,bias=False)
        self.para=Parameter(torch.Tensor(3,3))

    def forward(self,x):
        x1=self.l1(x)
        x2=self.l2(x1)
        return x2


mynet=TestNet()

for p in mynet.parameters():#在自定义网络模型时的parameter将会出现在网络模型的参数列表当中
    print(p)
    print('======================================================================')