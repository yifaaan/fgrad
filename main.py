import numpy as np
import struct
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import trange
from util import fetch_mnist


class BobNet(nn.Module):
    def __init__(self):
        super(BobNet, self).__init__()
        self.l1 = nn.Linear(28*28, 128, bias=False)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(128, 10, bias=False)
    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x


if __name__ == "__main__":
    # load data
    X_train, Y_train, X_test, Y_test = fetch_mnist()
    # plt.imsave("a.png", X_train[0], cmap='gray')

    # training
    model = BobNet()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
    plt.ylim(-0.1, 1.1)
    plt.legend()
    # plt.show()
    plt.close()


    # evaluation
    out =model(torch.tensor(X_test, dtype=torch.float32).reshape(-1, 28*28))
    cat = torch.argmax(out, dim=1)
    accuracy = (cat == torch.tensor(Y_test, dtype=torch.long)).float().mean()
    print('accuracy of pytorch model: ', accuracy)


    # **** NO MORE PYTORCH HEAR ****

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
    Y_test_pred = np.argmax(forward(X_test.reshape(-1, 28*28)), axis=1)
    accuracy = (Y_test_pred == Y_test).mean()
    print('accuracy of numpy model: ', accuracy)
# training

# evaluation