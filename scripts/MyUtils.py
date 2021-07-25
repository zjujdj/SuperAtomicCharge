import torch
import random
import numpy as np
from torch.utils.data.sampler import Sampler
import datetime
import torch.nn as nn
import os


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# 在每个epoch开始的时候, 自定义从数据集中取样本的策略
class DTISampler(Sampler):  # 采样器
    def __init__(self, weights, num_samples, replacement=True):
        weights = np.array(weights) / np.sum(weights)  # 返回一个矩阵，np.sum(weights) = 2,  weights之和等于1
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        # return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())
        retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights)
        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples


class EarlyStopping(object):
    def __init__(self,  mode='higher', patience=15, filename=None, tolerance=0.0):
        if filename is None:
            dt = datetime.datetime.now()
            filename = './model_save/early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.tolerance = tolerance
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        # return (score > prev_best_score)
        return score / prev_best_score > 1 + self.tolerance

    def _check_lower(self, score, prev_best_score):
        # return (score < prev_best_score)
        return prev_best_score / score > 1 + self.tolerance

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)  # 保存网络中的参数, 速度快，占空间少, 以字典格式存储

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])


class MyLoss(nn.Module):
    def __init__(self, alph):
        super(MyLoss, self).__init__()
        self.alph = alph

    def forward(self, input, target):
        sum_xy = torch.sum(torch.sum(input * target))
        sum_x = torch.sum(torch.sum(input))
        sum_y = torch.sum(torch.sum(target))
        sum_x2 = torch.sum(torch.sum(input * input))
        sum_y2 = torch.sum(torch.sum(target * target))
        n = input.size()[0]
        pcc = (n * sum_xy - sum_x * sum_y) / torch.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        return self.alph*(1-torch.abs(pcc)) + (1-self.alph)*torch.nn.functional.mse_loss(input, target)