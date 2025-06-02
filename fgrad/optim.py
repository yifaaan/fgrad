import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params
    
    
    def zero_grad(self):
        for t in self.params:
            t.grad = None

class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params)
        self.lr = lr

    # update the weights
    def step(self):
        for t in self.params:
            t.data -= self.lr * t.grad

    

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = [np.zeros_like(t.data) for t in self.params]
        self.v = [np.zeros_like(t.data) for t in self.params]
        self.t = 0
    
    def step(self):
        # 在梯度历史上计算
        self.t += 1
        for i, t in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * t.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.square(t.grad)
            m_hat = self.m[i] / (1 - self.b1**self.t)
            v_hat = self.v[i] / (1 - self.b2**self.t)
            t.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        