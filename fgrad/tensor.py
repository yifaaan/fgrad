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
  
  def __str__(self):
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
      if g.shape != t.data.shape:
        print("grad shape must match tensor shape in %r, %r != %r" % (self._ctx.arg, g.shape, t.data.shape))
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
    ctx = arg(self, *x)
    # self.data `op` *x
    ret = Tensor(arg.forward(ctx, self.data,*[t.data for t in x]))
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
    # inspired byhttps://gregorygundersen.com/blog/2020/02/09/log-sum-exp/?utm_source=chatgpt.com [6]
    def logsumexp(x):
      c = np.max(x,axis=1,keepdims=True)
      return c + np.log(np.exp(x - c).sum(axis=1,keepdims=True))
    output = input - logsumexp(input)
    ctx.save_for_backward(output)
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    output, = ctx.saved_tensors
    row_sum = grad_output.sum(axis=1,keepdims=True)
    return grad_output - np.exp(output) * row_sum
register("logsoftmax", LogSoftmax)