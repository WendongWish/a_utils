#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 21-09-09 0009 17:19
# @Author  : Hy Wang
# @Software: PyCharm

import torch
from thop import profile, clever_format

if __name__ == '__main__':
    image = torch.randn(1, 3, 224, 224).cuda()
    # net = MFM_Net().cuda()  # 输入网络名

    flops, params = profile(net, inputs=(image,))
    flops, params = clever_format([flops, params], '%.6f')
    print(params, flops)
    """    
    floating point operations的缩写（s表复数），意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。
    """
