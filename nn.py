import numpy as np
def layer_init(m, h):
    ret = np.random.uniform(-1., 1., size=(m,h), dtype=np.float32) / np.sqrt(m*h)
    return ret

class SGD:
    def __init___(self, tensors, lr):
        self.tensors = tensors
        self.lr = lr
    
    # update the weights
    def step(self):
        for t in self.tensors:
            t.data -= self.lr * t.grad
