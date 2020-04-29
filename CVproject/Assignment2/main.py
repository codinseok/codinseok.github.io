import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import TrainData, TestData


class ThreeLayersNet(nn.Module):
    def __init__(self):
        super(ThreeLayersNet, self).__init__()
        self.lin1 = nn.Linear(2, 3)
        self.lin2 = nn.Linear(3, 3)
        self.lin3 = nn.Linear(3, 2)

    def forward(self, x):
        out = F.relu(self.lin1(x))
        out = F.relu(self.lin2(out))
        out = self.lin3(out)
        return out


def train(epoch, data_loader):
    model.train()
    loss_total = 0
    for iteration, batch in enumerate(data_loader, 1):
        input_data = batch[0]
        gt = batch[1]
        gt = torch.squeeze(gt)
        gt = gt.long()
        # gt = torch.unsqueeze(gt, 1)
        out = model(input_data.float())
        loss = loss_fn(out, gt)

        loss.backward()
        optimizer.step()

        loss_total += loss.item()

        print("Epoch[{}]({}/{}): Loss = {:.6f}".format(epoch, iteration, len(data_loader), loss.item()))

    print('avg loss = %f' % ((loss_total / len(data_loader))))
    log('Epoch[%d] : Avg loss = %f\n' % (epoch, (loss_total / len(data_loader))), logfile)


def test(epoch, data_loader):
    model.eval()
    acc_sum = 0
    n = 0
    with torch.no_grad():
        for iteration, batch in enumerate(data_loader, 1):
            input_data = batch[0]
            gt = batch[1]
            gt = torch.squeeze(gt)
            optimizer.zero_grad()

            out = model(input_data.float())
            correct = accuracy(out, gt)
            acc_sum += correct
            n += 1
        acc = acc_sum / n
    return acc


def log(text, LOGGER_FILE):
    with open(LOGGER_FILE, 'a') as f:
        f.write(text)
        f.close()


def accuracy(pred, target):
    m = nn.Softmax(dim=1)
    pred_tag = m(pred)
    if pred_tag[0][0] > 0.5:
        pred_tag = 0
    else:
        pred_tag = 1
    correct_sum = (pred_tag == target).sum().float()
    # acc = correct_sum / target.shape[0]
    # acc = torch.round(acc * 100)

    return correct_sum


if __name__ == '__main__':

    model = ThreeLayersNet()
    logfile = 'logfile.txt'
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75)

    train_set = TrainData()
    train_data_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True, pin_memory=True)
    test_set = TestData()
    test_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, pin_memory=True)

    loss_fn = nn.CrossEntropyLoss()
    test_acc = np.array([])
    for epoch in range(1000):
        train(epoch, train_data_loader)

        if epoch % 10 == 0:
            result = test(epoch, test_data_loader)
            test_acc = np.append(test_acc, result)
            # test(epoch, test_data_loader)

        scheduler.step()

    plt.plot(test_acc)
    plt.show()

