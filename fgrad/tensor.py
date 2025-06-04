import numpy as np
from functools import partialmethod

# **** start with three base classes ****

class Tensor:
  def __init__(self, data):
    if type(data) != np.ndarray:
      print("error constructing tensor with %r" % data)
      assert False
    self.data = data
    self.grad = None
    # internal variables used for autograd graph construction
    self._ctx = None
  
  def __repr__(self):
    return "Tensor %r with grad %r" % (self.data, self.grad)

  def backward(self, allow_fill=True):
    if self._ctx == None:
      return
    if self.grad is None and allow_fill:
      assert self.data.size == 1
      self.grad = np.ones_like(self.data)
    
    assert(self.grad is not None)
    
    grads = self._ctx.backward(self._ctx, self.grad)
    if len(self._ctx.parents) == 1:
      grads = [grads]
    for t,g in zip(self._ctx.parents, grads):
      if g is None:
        continue
      if g.shape != t.data.shape:
        print("grad shape must match tensor shape in %r, %r != %r" % (self._ctx, g.shape, t.data.shape))
        assert False
      t.grad = g
      t.backward(False)
  
  def mean(self):
      div = Tensor(np.array([1/self.data.size]))
      return self.sum().mul(div)


# encode the operation history and compute gradients.
class Function:
  def __init__(self, *tensors):
    self.parents = tensors
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)
  
  def apply(self, arg, *x):
    # apply called by Function
    if type(arg) == Tensor:
      op = self
      x = [arg] + list(x)
    else:
      # apply called by Tensor
      op = arg
      x = [self] + list(x)
    # for saving the parents
    ctx = op(*x)
    ret = Tensor(op.forward(ctx, *[t.data for t in x]))
    ret._ctx = ctx
    return ret

def register(name, fxn):
  # bind fxn.apply to Tensor.name
  # def apply(self, fxn, *x):
    # self  → Tensor
    # fxn   → Function （Dot / ReLU / …）
    # *x    → input Tensors
  setattr(Tensor, name, partialmethod(fxn.apply, fxn))


# **** implement a few functions ****

class ReLU(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.maximum(input, 0)
  
  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output.copy()
    grad_input[input < 0] = 0
    return grad_input
register("relu", ReLU)


class Dot(Function):
  @staticmethod
  def forward(ctx, input, weight):
    ctx.save_for_backward(input, weight)
    return input.dot(weight)
  
  @staticmethod
  def backward(ctx, grad_output):
    # input: (N, in), weight: (in, out)
    # grad_output: (N, out), grad_input: (N, in), grad_weight: (in, out)
    # X @ W.T = out
    # get children
    input, weight = ctx.saved_tensors
    # use dims to judge
    grad_input = grad_output.dot(weight.T)
    grad_weight = grad_output.T.dot(input).T
    return grad_input, grad_weight
register("dot", Dot)

class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x * y
  
  @staticmethod
  def backward(ctx, grad_output):
    x, y = ctx.saved_tensors
    return y * grad_output, x * grad_output
register("mul", Mul)

class Sum(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.array([input.sum()])
  
  @staticmethod
  def backward(ctx, grad_ouput):
    input, = ctx.saved_tensors
    return grad_ouput * np.ones_like(input)
register("sum", Sum)

class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x + y
  
  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output
register("add", Add)


class LogSoftmax(Function):
  @staticmethod
  def forward(ctx, input):
    # logsumexp to avoid overflow
    # inspired by https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/?utm_source=chatgpt.com [6]
    def logsumexp(x):
      c = np.max(x,axis=1,keepdims=True)
      return c + np.log(np.exp(x - c).sum(axis=1,keepdims=True))
    output = input - logsumexp(input)
    ctx.save_for_backward(output)
    # the output is not probabilities bug logprobabilities(which are between -INF and 0)
    # It’s often paired with loss functions like Negative Log Likelihood (NLL) Loss or Cross-Entropy Loss.
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    output, = ctx.saved_tensors
    row_sum = grad_output.sum(axis=1,keepdims=True)
    return grad_output - np.exp(output) * row_sum
register("logsoftmax", LogSoftmax)

class Conv2D(Function):
  @staticmethod
  def forward(ctx, x, w):
    ctx.save_for_backward(x, w)
    cout, cin, H, W = w.shape
    # ret: (N, cout, H', W')
    ret = np.zeros((x.shape[0], cout, x.shape[2] - (H - 1), x.shape[3] - (W - 1)), dtype=w.dtype)
    # inspired by https://cs231n.github.io/convolutional-networks/?utm_source=chatgpt.com
    for j in range(H):
      for i in range(W):
        # evaluate the convolution at (j, i)
        tw = w[:, :, j, i]
        for Y in range(ret.shape[2]):
          for X in range(ret.shape[3]):
            tx = x[:, :, Y+j, X+i]
            ret[:, :, Y, X] += tx.dot(tw.T)
    
    return ret

  @staticmethod
  def backward(ctx, grad_output):
    # grad_w: (cout, cin, H, W)
    # grad_output: (N, cout, H', W') same as output's shape
    # inspired by https://coolgpu.github.io/coolgpu_blog/github/pages/2020/10/04/convolution.html
    input, weight = ctx.saved_tensors
    grad_input, grad_weight = np.zeros_like(input), np.zeros_like(weight)
    for Y in range(grad_output.shape[2]):
      for X in range(grad_output.shape[3]):
        # grad_output[:, :, Y, X],  (N, cout) 的张量,表示在输出特征图的 (Y, X) 位置上，损失对每个输出通道的梯度值
        for j in range(weight.shape[2]): # 卷积核的每个高度位置
          for i in range(weight.shape[3]): # 卷积核的每个宽度位置
            tx = input[:, :, Y+j, X+i]
            tw = weight[:, :, j, i] # 形状 (cout, cin)，这是卷积核在 (j,i) 位置所有通道的权重
            grad_input[:, :, Y+j, X+i] += grad_output[:, :, Y, X].dot(tw)
            grad_weight[:, :, j, i] += grad_output[:, :, Y, X].T.dot(tx)
    return grad_input, grad_weight
register("conv2d", Conv2D)

class Reshape(Function):
  @staticmethod
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    return x.reshape(shape)
  
  @staticmethod
  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    return grad_output.reshape(in_shape), None
register("reshape", Reshape)