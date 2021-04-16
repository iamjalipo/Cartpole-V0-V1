import torch

from torch import nn

class firstmodule(nn.Module):
    def __init__(self , num_inputs , num_classes , dropout_prob = 0.3):
        super(firstmodule , self).__init__
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1)
        )
