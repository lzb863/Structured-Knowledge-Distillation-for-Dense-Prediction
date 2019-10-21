# coding=utf-8

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from libs.net.sync_batchnorm import SynchronizedBatchNorm2d


class ASM(nn.Module):
    def __init__(self, num_classes, num_scale):
        super(ASM, self).__init__()
        # {s1,s2,s3} 不同尺度图片输入 共享同一个网络 net, 输出 {s1o,s2o,s3o}
        # 双线性插值到同一个尺度, 通道concate, 输入ASM模块中

        self.num_scale = num_scale
        self.num_classes = num_classes
        self.asm = nn.Sequential(
            nn.Conv2d(num_classes * num_scale, 512, 3, 1, padding=1), # relu ??
            nn.Dropout(0.5),
            nn.Conv2d(512, num_classes, 1, 1, padding=0),
        )



    def forward(self, raw_x):
        x = self.asm(raw_x)
        x = nn.functional.softmax(x, dim=1)
        x_scale = list(torch.split(x, 1, dim=1))
        raw_x_scale = list(torch.split(raw_x, self.num_classes, dim=1))
        s_map = torch.zeros_like(raw_x_scale[0])
        for ind, s in enumerate(x_scale):
            s_map = torch.add(torch.mul(s, raw_x_scale[ind]), s_map)
        s_map = torch.argmax(s_map, dim=1)
        return s_map



if __name__ == '__main__':
    x = torch.rand((1,9,5,5))
    asm = ASM(3, 3)
    s_map = asm(x)

    z = 1