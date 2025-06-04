# from nn import BobNet
import numpy as np
from tqdm import trange
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fgrad.utils import fetch_mnist, layer_init_uniform
from fgrad.tensor import Tensor
import fgrad.optim as optim


class FBobNet:
  def __init__(self):
    self.l1 = Tensor(layer_init_uniform(28*28, 128))
    self.l2 = Tensor(layer_init_uniform(128, 10))
  
  def forward(self, x):
    x = x.dot(self.l1).relu()
    return x.dot(self.l2).logsoftmax()

class FConvNet:
  def __init__(self):
      self.chans = 4
      self.c1 = Tensor(layer_init_uniform(self.chans, 1, 3, 3)) 
      # c1's output  (28 - 3 + 1) * (28 - 3 + 1) * 4
      self.l1 = Tensor(layer_init_uniform(26*26*self.chans, 128))
      self.l2 = Tensor(layer_init_uniform(128, 10))

  def forward(self, x):
    x.data = x.data.reshape((-1, 1, 28, 28))
    x = x.conv2d(self.c1).reshape(Tensor(np.array((-1, 26*26*self.chans)))).relu()
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()





X_train, Y_train, X_test, Y_test = fetch_mnist()
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

if os.getenv("CONV") == "1":
  model = FConvNet()
  optim = optim.Adam([model.c1, model.l1, model.l2], lr=0.001)
  steps = 400
else:
  model = FBobNet()
  optim = optim.Adam([model.l1, model.l2], lr=0.001)
  steps = 1000




# training
BS = 128

losses, accuracies = [], []
for i in (t := trange(steps)):
  samp = np.random.randint(0, X_train.shape[0], BS)
  x = Tensor(X_train[samp].reshape(-1, 28*28))
  Y = Y_train[samp]
  # create one-hot matrix
  y = np.zeros((BS, 10), dtype=np.float32)
  # set true label to -1:one-hot
  y[np.arange(BS), Y] = -10.0
  y = Tensor(y)

  # network
  out = model.forward(x)

  # NLL loss function
  
  loss = out.mul(y).mean()
  loss.backward()
  optim.step()
  
  cat = np.argmax(out.data, axis=1)
  accuracy = (cat == Y).mean()


  # printing
  loss = loss.data
  losses.append(loss)
  accuracies.append(accuracy)
  t.set_description("loss %.2f, accuracy %.2f" % (loss, accuracy))


def numpy_eval():
  Y_test_pred_out = model.forward(Tensor(X_test.reshape(-1, 28*28))) # [N, 10]
  Y_test_pred = np.argmax(Y_test_pred_out.data, axis=1)
  accuracy = (Y_test_pred == Y_test).mean()
  return accuracy

print("test set accuracy is %f" % numpy_eval())