import torch

from torch import nn

s = nn.Sequential(
    nn.Linear(2,5),
    nn.ReLU(),
    nn.Linear(5,7),
    nn.ReLU(),
    nn.Linear(7,10),
    nn.Dropout(p = 0.3),
    nn.Softmax(dim=1)
)

print(s(torch.Tensor([[2,5]])))