import torch

x1=torch.tensor([[1,2,3.],[2,2,3]])
x2=torch.tensor([[4,5],[4.,5],[4,5]])
out=torch.mm(x1,x2)
print(out)