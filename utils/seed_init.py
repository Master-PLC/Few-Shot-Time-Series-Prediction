#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :seed_init.py
@Description  :
@Date         :2021/11/08 14:59:21
@Author       :Arctic Little Pig
@Version      :1.0
'''

import random

import numpy as np
import torch


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
