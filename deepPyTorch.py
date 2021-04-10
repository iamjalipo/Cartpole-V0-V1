import torch

a = torch.tensor([1,2,3])

x = a.sum()
print(x.item())