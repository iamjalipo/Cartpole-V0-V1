import torch

from torch import nn

class firstmodule(nn.Module):
    def __init__(self , num_inputs , num_classes , dropout_prob = 0.3):
        super(firstmodule , self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1)
        )
    def forward(self , x):
        return self.pipe(x)

if __name__ == "__main__":
    net = firstmodule(num_inputs=2, num_classes=3)
    v = torch.FloatTensor([[2, 3]])
    out = net(v)
    print(net)
    print(out)