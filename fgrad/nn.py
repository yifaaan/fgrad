import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def layer_init(m, h):
    ret = np.random.uniform(-1., 1., size=(m,h), dtype=np.float32) / np.sqrt(m*h)
    return ret

class BobNet(nn.Module):
    def __init__(self):
        super(BobNet, self).__init__()
        self.l1 = nn.Linear(28*28, 128, bias=False)
        self.l2 = nn.Linear(128, 10, bias=False)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = nn.LogSoftmax(dim=1)(x)
        return x
