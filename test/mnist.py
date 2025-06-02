from nn import BobNet
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import trange
from util import fetch_mnist
from fgrad import Tensor, optim


def use_torch():
    # load data
    X_train, Y_train, X_test, Y_test = fetch_mnist()
    # plt.imsave("a.png", X_train[0], cmap='gray')
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    # training
    model = BobNet()
    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.NLLLoss() # 接收「对数概率」和标签
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = SGD(model.parameters(), lr=0.001)
    # optimizer = Adam(model.parameters(), lr=0.001)
    BS = 32
    losses, accuracies = [], []
    for i in (t := trange(1000)):
        samp = np.random.randint(0, X_train.shape[0], BS)
        X = torch.tensor(X_train[samp], dtype=torch.float32).reshape(-1, 28*28)
        Y = torch.tensor(Y_train[samp], dtype=torch.long)
        out = model(X)
        cat = torch.argmax(out, dim=1)
        # print(cat)
        accuracy = (cat == Y).float().mean()
        loss = loss_func(out, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss, accuracy = loss.item(), accuracy.item()
        losses.append(loss)
        accuracies.append(accuracy)
        t.set_description("loss %.2f, accuracy %.2f" % (loss, accuracy))

    # draw loss and accuracy curve
    plt.figure()
    plt.plot(losses, label='Loss')
    plt.xlabel('Iteration')
    plt.plot(accuracies, label='Accuracy')
    # plt.ylim(-0.1, 1.1)
    plt.legend()
    # plt.show()
    plt.close()


    # evaluation
    out =model(torch.tensor(X_test, dtype=torch.float32).reshape(-1, 28*28))
    cat = torch.argmax(out, dim=1)
    accuracy = (cat == torch.tensor(Y_test, dtype=torch.long)).float().mean()
    print('accuracy of pytorch model: ', accuracy)


    # **** NO MORE PYTORCH HEAR ****
    plt.imsave("grad1.png", model.l1.weight.grad)
    plt.imsave("grad2.png", model.l2.weight.grad)
    plt.close()

    # init the nextwork
    l1 = np.zeros((128, 28*28), dtype=np.float32)
    l2 = np.zeros((10, 128), dtype=np.float32)
    l1[:] = model.l1.weight.detach().numpy()
    l2[:] = model.l2.weight.detach().numpy()

    def forward(x):
        x = x @ l1.T
        # relu
        x = np.maximum(x, 0)
        x = x @ l2.T
        return x
    
    # use cross entropy loss
    Y_test_pred_out = forward(X_test.reshape(-1, 28*28)) # [N, 10]
    # softmax
    p = np.exp(Y_test_pred_out) / np.sum(np.exp(Y_test_pred_out), axis=1, keepdims=True)
    
    log_p = np.log(p)
    ret = -log_p[np.arange(Y_test.shape[0]), Y_test]
    loss = np.mean(ret)
    print('loss of numpy model: ', loss)
    G = 16
    grid = sorted(list(zip(ret, np.arange(ret.shape[0]))), reverse=True)[:G*G]
    X_bad = X_test[[id for (_, id) in grid]]

    img = np.concatenate(X_bad.reshape((G, 28*G, 28)), axis=1)
    plt.imsave("grid.png", img)
    plt.close()

    # Y_test_pred = np.argmax(forward(X_test.reshape(-1, 28*28)), axis=1)
    # accuracy = (Y_test_pred == Y_test).mean() * 100
    # print('accuracy of numpy model: ', accuracy)

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
        loss = x.data
        
        cat = np.argmax(x_l2.data, axis=1)
        accuracy = (cat == Y).mean()

        optimizer = optim.Adam([l1, l2], lr)
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