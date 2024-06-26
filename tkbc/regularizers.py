# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass

class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]

class N4(Regularizer):
    def __init__(self, weight: float):
        super(N4, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 4)
        return norm / factors[0].shape[0]

class N5(Regularizer):
    def __init__(self, weight: float):
        super(N5, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 4)
        return norm / factors[0].shape[0]

class Lambda3(Regularizer):
    def __init__(self, weight: float):
        super(Lambda3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        ddiff = factor[1:] - factor[:-1]
        rank = int(ddiff.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)

class Lambda3_two(Regularizer):
    def __init__(self, weight: float):
        super(Lambda3_two, self).__init__()
        self.weight = weight
    def forward(self, factor):
        tot = 0
        if factor is not None:
            for f in factor:
                rank = int(f.shape[1] / 2)
                ddiff = f[1:] - f[:-1]
                diff = torch.sqrt((ddiff[:, :rank]**2 + ddiff[:, rank:]**2))**4
                tot = tot + self.weight * (torch.sum(diff))
            return tot / factor[0].shape[0]
        return 0

class Lambda4(Regularizer):
    def __init__(self, weight: float):
        super(Lambda4, self).__init__()
        self.weight = weight

    def forward(self, factor):
        ddiff = factor[1:] - factor[:-1]
        rank = int(ddiff.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**4
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)

class Linear3(Regularizer):
    def __init__(self, weight: float):
        super(Linear3, self).__init__()
        self.weight = weight

    def forward(self, factor, W):
        rank = int(factor.shape[1] / 2)
        ddiff = factor[1:] - factor[:-1] - W.weight[:rank*2].t()
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)