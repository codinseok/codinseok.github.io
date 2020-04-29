import numpy as np
from torch.utils.data import Dataset
import torch


class TrainData(Dataset):
    def __init__(self):
        data = np.loadtxt(open('data.csv', 'r'), delimiter=',')
        gt = np.loadtxt(open('gt.csv', 'r'), delimiter=',')
        self.data = data
        self.gt = gt

    def __getitem__(self, index):

        x = self.data[index]
        gt = self.gt[index]
        # gt_tmp = self.gt[index]
        # gt = np.zeros((1, 2), np.float32)
        #
        # if gt_tmp == 0:
        #     gt[0, 1] = 1
        #     gt[0, 0] = gt_tmp
        # else:
        #     gt[0, 1] = 0
        #     gt[0, 0] = gt_tmp

        x = torch.tensor(x)
        gt = torch.tensor(gt)
        return x, gt

    def __len__(self):
        return int(len(self.gt))


class TestData(Dataset):
    def __init__(self):
        data = np.loadtxt(open('test.csv', 'r'), delimiter=',')
        gt = np.loadtxt(open('test_gt.csv', 'r'), delimiter=',')
        self.data = data
        self.gt = gt

    def __getitem__(self, index):

        x = self.data[index]
        gt = self.gt[index]

        x = torch.tensor(x)
        gt = torch.tensor(gt)
        return x, gt

    def __len__(self):
        return int(len(self.gt))


if __name__ == '__main__':
    trainset = TrainData()
    input_all = trainset.__getitem__(10)
    print(input_all)
