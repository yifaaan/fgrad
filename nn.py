import numpy as np
import torch.nn as nn

def layer_init(m, h):
    ret = np.random.uniform(-1., 1., size=(m,h), dtype=np.float32) / np.sqrt(m*h)
    return ret

class BobNet(nn.Module):
    def __init__(self):
        super(BobNet, self).__init__()
        self.l1 = nn.Linear(28*28, 128, bias=False)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(128, 10, bias=False)
    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x
