import numpy as np
import torch
import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fgrad.tensor import Tensor, Conv2D

x_init = np.random.randn(1,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

# def test_fgrad():
#   x = Tensor(x_init)
#   W = Tensor(W_init)
#   m = Tensor(m_init)
#   out = x.dot(W)
#   outr = out.relu()
#   outl = outr.logsoftmax()
#   outm = outl.mul(m)
#   outa = outm.add(m)
#   outx = outa.sum()
#   outx.backward()
#   return outx.data, x.grad, W.grad

# def test_pytorch():
#   x = torch.tensor(x_init, requires_grad=True)
#   W = torch.tensor(W_init, requires_grad=True)
#   m = torch.tensor(m_init)
#   out = x.matmul(W)
#   outr = out.relu()
#   outl = torch.nn.functional.log_softmax(outr, dim=1)
#   outm = outl.mul(m)
#   outa = outm.add(m)
#   outx = outa.sum()
#   outx.backward()
#   return outx.detach().numpy(), x.grad, W.grad

# for x,y in zip(test_fgrad(), test_pytorch()):
# 	print(x,y)
# 	np.testing.assert_allclose(x, y, atol=1e-6)
  
class TestFgrad(unittest.TestCase):
  def test_backward_pass(self):
    def test_fgrad():
      x = Tensor(x_init)
      W = Tensor(W_init)
      m = Tensor(m_init)
      out = x.dot(W).relu() # (1, 3)
      out = out.logsoftmax() # (1, 3)
      out = out.mul(m).add(m) # (1, 3)
      out = out.sum() # (1,)
      out.backward()
      return out.data, x.grad, W.grad
    
    def test_pytorch():
      x = torch.tensor(x_init, requires_grad=True)
      W = torch.tensor(W_init, requires_grad=True)
      m = torch.tensor(m_init)
      out = x.matmul(W).relu() # (1, 3)
      out = torch.nn.functional.log_softmax(out, dim=1) # (1, 3)
      out = out.mul(m).add(m) # (1, 3)
      out = out.sum() # (1,)
      out.backward()
      return out.detach().numpy(), x.grad, W.grad
    
    for x,y in zip(test_fgrad(), test_pytorch()):
      np.testing.assert_allclose(x, y, atol=1e-5)

  def test_conv2d(self):
    x = torch.randn((5, 2, 10, 7))
    w = torch.randn((4, 2, 3, 3))

    out = torch.nn.functional.conv2d(x, w)
    ret = Conv2D.apply(Conv2D, Tensor(x.numpy()), Tensor(w.numpy()))
    np.testing.assert_allclose(ret.data, out.numpy(), atol=1e-5)

if __name__ == "__main__":
    unittest.main()
        