# from nn import BobNet
import numpy as np
from tqdm import trange
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fgrad.utils import fetch_mnist, layer_init_uniform
from fgrad.tensor import Tensor
from fgrad.optim import Adam, SGD



X_train, Y_train, X_test, Y_test = fetch_mnist()
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
# training
l1 = Tensor(layer_init_uniform(28*28, 128))
l2 = Tensor(layer_init_uniform(128, 10))
lr = 0.01
BS = 128
optimizer = Adam([l1, l2], lr, weight_decay=0.01)
# optimizer = SGD([l1, l2], lr)

losses, accuracies = [], []
for i in (t := trange(1000)):
    samp = np.random.randint(0, X_train.shape[0], BS)
    X = Tensor(X_train[samp].reshape(-1, 28*28))
    Y = Y_train[samp]
    # create one-hot matrix
    y = np.zeros((BS, 10), dtype=np.float32)
    # set true label to -1:one-hot
    y[np.arange(BS), Y] = -1
    y = Tensor(y)

    x = X.dot(l1)
    x = x.relu()
    x = x_l2 = x.dot(l2)
    x = x.logsoftmax()

    x = x.mul(y)
    x = x.mean()
    x.backward()
    loss = x.data[0]
    
    cat = np.argmax(x_l2.data, axis=1)
    accuracy = (cat == Y).mean()

    
    optimizer.step()
    optimizer.zero_grad()

    # l1.data -= lr * l1.grad
    # l2.data -= lr * l2.grad

    
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description("loss %.2f, accuracy %.2f" % (loss, accuracy))

       

def forward(x):
    x = x.dot(l1.data)
    x = np.maximum(x, 0)
    x = x.dot(l2.data)
    return x

def numpy_eval():
    Y_test_pred_out = forward(X_test.reshape(-1, 28*28)) # [N, 10]
    Y_test_pred = np.argmax(Y_test_pred_out, axis=1)
    accuracy = (Y_test_pred == Y_test).mean()
    return accuracy

print("test set accuracy is %f" % numpy_eval())