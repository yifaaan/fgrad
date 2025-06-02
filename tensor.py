import numpy as np
from functools import partialmethod

class Context:
  def __init__(self, arg, *tensors):
    self.arg = arg
    self.parents = tensors
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

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
    
    grads = self._ctx.arg.backward(self._ctx, self.grad)
    if len(self._ctx.parents) == 1:
      grads = [grads]
    for t,g in zip(self._ctx.parents, grads):
      if g.shape != t.data.shape:
        print("grad shape must match tensor shape in %r, %r != %r" % (self._ctx.arg, g.shape, t.data.shape))
        assert False
      t.grad += g
      t.backward(False)

class Function:
  def apply(self, arg, *x):
    ctx = Context(self, arg, *x)
    ret = Tensor(arg.forward(ctx, arg,*[t.data for t in x]))
    ret._ctx = ctx
    return ret

def register(name, fxn):
  setattr(Tensor, name, partialmethod(fxn.apply, fxn))

class ReLU(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.maximum(input, 0)
  
  @staticmethod
  def backward(ctx, grad_output):
    input = ctx.saved_tensors
    grad_input = grad_output.clone()
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
    # X @ W.T = out
    # get children
    input, weight = ctx.saved_tensors
    # use dims to judge
    grad_input = grad_output.dot(weight)
    grad_weight = grad_output.T.dot(input)
    return grad_input, grad_weight
register("dot", Dot)

class Sum(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return input.sum()
  
  @staticmethod
  def backward(ctx, grad_ouput):
    input = ctx.saved_tensors
    return grad_ouput * np.ones_like(input)
register("sum", Sum)
