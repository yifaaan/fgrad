# from nn import BobNet
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import trange
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fgrad.tensor import Tensor
from fgrad.optim import Adam, SGD


import struct
import numpy as np

def fetch_mnist():

    def fetch(url):
        import requests, gzip, os, hashlib, numpy
        fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
        if os.path.isfile(fp):
            with open(fp, 'rb') as f:
                data = f.read()
        else:
            with open(fp, 'wb') as f:
                data = requests.get(url).content
                f.write(data)
        return numpy.frombuffer(gzip.decompress(data), dtype=np.uint8)

    # 16 字节的文件头, 前 4 字节是 magic number, 后面依次为图像数量, 行数, 列数 (都用大端表示)
    file_data = fetch('https://storage.googleapis.com/kaggle-data-sets/102285/242592/upload/train-images-idx3-ubyte.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250531%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250531T051117Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=23ffd289ca39356b793279e5b8de4b0ae86c48870508f96a4482cd24eb477909581ef45417ecfa4fb89fa259c69cf3d996b8d34c1120203488bb642c9b895e7868ba028397bc4c82afebcce6d52c92f0dc37e64778e50459e4ff684fe13a5c9847410c064e3b8d98bd77a5bbc81313ff42033935462ce8dc3863191d57b4a084a1bd1314d06d1c524d11a84f7de6fdce2e35b0fc6dbec043bd3d1d512fe8823122da6f55d57e884e786aca405a1094f1680e5b25d68025ba13ae646352040fc680a2f32be7d28ce75b4045460c100e814715289635da69c04cb66bf3d1095b672cf2ed1dca465dbff8a1855d3961c6218261ff3e50308c67f2f62daa82cd745b')
    _, num_images, rows, cols = struct.unpack(">IIII", file_data[:0x10])
    X_train = file_data[0x10:].reshape(-1, rows, cols)
    file_data = fetch('https://storage.googleapis.com/kaggle-data-sets/102285/242592/upload/train-labels-idx1-ubyte.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250531%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250531T051144Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3488764f5aa676f7591f2b2c14fb917b5038a79e09dca7fecdaf95e2fe8635908783047939cebaf98d769044033dccee5b227a7ad4b7cb2b03c6d9cf42cc0c5abacf01313ddbc19967b523c103077de72ac5f502d2a511bc7ab5c467b1d71f99e885a6fa0568610084ebea24a8ceeff5a090724beb9748335837f6b0761a9c9f53821421f1848c21dd4424efe3589f632b60a0d9fca72639de5aa68b19ab59f77d88d7c01785fb2a8844fb26afe1efcde46685374cb300edd1738242910d460adb495d4018830872dcfd41f756efcca0907998a644da8340ff6a434903124012d33052e18f6b12a7add7b4374979c6641aea13378dceba323e612c20902e5a45')
    _, num_labels = struct.unpack(">II", file_data[:0x8])
    Y_train = file_data[0x8:]
    file_data = fetch('https://storage.googleapis.com/kaggle-data-sets/102285/242592/upload/t10k-images-idx3-ubyte.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250531%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250531T045831Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=39f58cb3a2e8034659d5a4534eddd606e970c78e2063b61976ebea0efc5afa9edee8db6844fabca8a0461aa24e9121060cbe889be083b58ff3c4c1635af3b35bb4e9ab47b273fb8764b481f3640689a34d279e810653fadd94299f54a5a72846f0a5d7d904543243c21781c17a2ef3b7ac6e3bf9c882377aa94677dc50283b96fbccd721614a108dd83656563b28dccc3280c3e034bcab219a56a474fdf66a9a1f0e9662b105cb5c74a1c1f67aaa06ea7ff50297a53551575b742cb3f4c682a2614d534f3eb690130d12547b2ce752971ebe618f8ef8949f43d07b22ce4385c427ce6852f9b281582fdbefc2769c3b6d45110c633d1940cc017240be668c255b')
    _, num_images, rows, cols = struct.unpack(">IIII", file_data[:0x10])
    X_test = file_data[0x10:].reshape(-1, rows, cols)
    file_data = fetch('https://storage.googleapis.com/kaggle-data-sets/102285/242592/upload/t10k-labels-idx1-ubyte.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250531%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250531T045908Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=a28f7a0eb3d41cf0aeb47d9755eefa259b5c35a6e1e52be4913873cdfe451b5695d0ae1bf80b52e49923decc8443b224f63492c97a6dca211620882421d18f9d32baa1e3dbc01dc899a4260c582e8be09fb66e2fd05335ed54759c04d61baea0997f06cf7775b954c49e5df7fc7a7ff114e4c1554aef0390ba34c907acc73a47032cf4cf5d360c8c84e22321f06f21d003b48c789a72590914557b7a3cd6f0842005acb2db857a3115cb7e74529416e1926ed367d75eb632f4b59ae6b001f06fc97b5e772bb285c667928f697314c96c9a3c61b84aea4ae3eac7e091f6768a11cbd1dd68a0e577c655a7059e37354c3e589512cc10f4d951574980bae38b0f44')
    _, num_labels = struct.unpack(">II", file_data[:0x8])
    Y_test = file_data[0x8:]
    return X_train, Y_train, X_test, Y_test
    # print(X_test.shape)
    # print(Y_test.shape)
    # print(X_train.shape)
    # print(Y_train.shape)





# def use_torch():
#     # load data
#     X_train, Y_train, X_test, Y_test = fetch_mnist()
#     # plt.imsave("a.png", X_train[0], cmap='gray')
#     X_train = X_train.astype(np.float32) / 255.0
#     X_test = X_test.astype(np.float32) / 255.0
#     # training
#     model = BobNet()
#     loss_func = nn.CrossEntropyLoss()
#     # loss_func = nn.NLLLoss() # 接收「对数概率」和标签
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     # optimizer = SGD(model.parameters(), lr=0.001)
#     # optimizer = Adam(model.parameters(), lr=0.001)
#     BS = 32
#     losses, accuracies = [], []
#     for i in (t := trange(1000)):
#         samp = np.random.randint(0, X_train.shape[0], BS)
#         X = torch.tensor(X_train[samp], dtype=torch.float32).reshape(-1, 28*28)
#         Y = torch.tensor(Y_train[samp], dtype=torch.long)
#         out = model(X)
#         cat = torch.argmax(out, dim=1)
#         # print(cat)
#         accuracy = (cat == Y).float().mean()
#         loss = loss_func(out, Y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         loss, accuracy = loss.item(), accuracy.item()
#         losses.append(loss)
#         accuracies.append(accuracy)
#         t.set_description("loss %.2f, accuracy %.2f" % (loss, accuracy))

#     # draw loss and accuracy curve
#     plt.figure()
#     plt.plot(losses, label='Loss')
#     plt.xlabel('Iteration')
#     plt.plot(accuracies, label='Accuracy')
#     # plt.ylim(-0.1, 1.1)
#     plt.legend()
#     # plt.show()
#     plt.close()


#     # evaluation
#     out =model(torch.tensor(X_test, dtype=torch.float32).reshape(-1, 28*28))
#     cat = torch.argmax(out, dim=1)
#     accuracy = (cat == torch.tensor(Y_test, dtype=torch.long)).float().mean()
#     print('accuracy of pytorch model: ', accuracy)


#     # **** NO MORE PYTORCH HEAR ****
#     plt.imsave("grad1.png", model.l1.weight.grad)
#     plt.imsave("grad2.png", model.l2.weight.grad)
#     plt.close()

#     # init the nextwork
#     l1 = np.zeros((128, 28*28), dtype=np.float32)
#     l2 = np.zeros((10, 128), dtype=np.float32)
#     l1[:] = model.l1.weight.detach().numpy()
#     l2[:] = model.l2.weight.detach().numpy()

#     def forward(x):
#         x = x @ l1.T
#         # relu
#         x = np.maximum(x, 0)
#         x = x @ l2.T
#         return x
    
#     # use cross entropy loss
#     Y_test_pred_out = forward(X_test.reshape(-1, 28*28)) # [N, 10]
#     # softmax
#     p = np.exp(Y_test_pred_out) / np.sum(np.exp(Y_test_pred_out), axis=1, keepdims=True)
    
#     log_p = np.log(p)
#     ret = -log_p[np.arange(Y_test.shape[0]), Y_test]
#     loss = np.mean(ret)
#     print('loss of numpy model: ', loss)
#     G = 16
#     grid = sorted(list(zip(ret, np.arange(ret.shape[0]))), reverse=True)[:G*G]
#     X_bad = X_test[[id for (_, id) in grid]]

#     img = np.concatenate(X_bad.reshape((G, 28*G, 28)), axis=1)
#     plt.imsave("grid.png", img)
#     plt.close()

#     # Y_test_pred = np.argmax(forward(X_test.reshape(-1, 28*28)), axis=1)
#     # accuracy = (Y_test_pred == Y_test).mean() * 100
#     # print('accuracy of numpy model: ', accuracy)

def layer_init(m, h):
    ret = np.random.uniform(-1., 1., size=(m,h)).astype(np.float32) / np.sqrt(m*h)
    return ret

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  X_train = X_train.astype(np.float32) / 255.0
  X_test = X_test.astype(np.float32) / 255.0
  # training
  l1 = Tensor(layer_init(28*28, 128))
  l2 = Tensor(layer_init(128, 10))
  lr = 0.01  # 增加学习率
  BS = 128
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

      optimizer = Adam([l1, l2], lr)
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