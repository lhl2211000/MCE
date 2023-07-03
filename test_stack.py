import torch

# x1=torch.tensor([1,2.])
#
# x2=torch.tensor([11,22.])
#
# x3=torch.tensor([111,222.])

x1=torch.rand(128,100)

x2=torch.rand(128,100)

x3=torch.rand(128,100)

l1=[x1,x2,x3]
# print(l1)
x4=torch.stack(l1,dim=2)
print(x4.shape)
# print(x4.mean(dim=1))