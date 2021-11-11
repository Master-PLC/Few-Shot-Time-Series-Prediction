#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :metrics.py
@Description  :
@Date         :2021/11/08 21:25:47
@Author       :Arctic Little Pig
@Version      :1.0
'''

import os
import pickle as pkl
import time

import numpy as np
import pandas as pd
from numpy.core.fromnumeric import mean
from sklearn.metrics import mean_absolute_error


def compute_rmse(dataA, dataB):
    length = len(dataA)
    rmse = np.sqrt(np.sum([(a - b)**2 for a, b in zip(dataA, dataB)])/length)
    return rmse

def compute_rmse2(dataA, dataB):
    """ RMSE """
    t1 = np.sum((dataA - dataB) **2) / np.size(dataB)
    return np.sqrt(t1)

def iter_list(item, nums):
    return iter([item for _ in range(nums)])

def get_acc2(data1:np.ndarray, data2:np.ndarray)->float:
    acc_list = []
    for a, b in zip(data1, data2):
        if a < 0:
            acc_list.append(0)
        elif max(a, b)==0:
            pass
        else:
            acc_list.append(min(a, b) / max(a, b))
    return sum(acc_list) / len(acc_list)

def get_acc(y_pred, y_true):
    acc_list = []
    y_p = y_pred.reshape(-1)
    y_t = y_true.reshape(-1)
    for a, b in zip(y_p, y_t):
        if a < 0:
            acc_list.append(0)
        elif max(a, b)==0:
            pass
        else:
            acc_list.append(min(a, b) / max(a, b))
    return sum(acc_list) / len(acc_list)

def generate_header(params_dict:dict)->str:
    header1 = "======== Configuration ========\n"
    header2 = ''
    for key in sorted(params_dict.keys(), key=len):
        header2 += "{} : {}\n".format(key,params_dict[key])
    
    header3="===============================\n"
    header = header1 + header2 + header3
    return header

def nd(y_pred, y_true):
    """ Normalized deviation"""
    t1 = np.sum(abs(y_pred-y_true)) / np.size(y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    return t1 / t2

def SMAPE(y_pred, y_true):
    s = 0
    y_p = y_pred.reshape(-1)
    y_t = y_true.reshape(-1)
    for a, b in zip(y_p, y_t):
        if abs(a) + abs(b) == 0:
            s += 0
        else:
            s += 2 * abs(a-b) / (abs(a) + abs(b))
    return s / np.size(y_true)

def nrmse(y_pred, y_true):
    """ Normalized RMSE"""
    t1 = np.linalg.norm(y_pred - y_true)**2 / np.size(y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    return np.sqrt(t1) / t2

def get_metrics(y_pred, y_true):
    
    index_d = {}
    index_d['acc'] = get_acc(y_pred, y_true)
    index_d['rmse'] = compute_rmse2(y_pred, y_true)
    index_d['nrmse'] = nrmse(y_pred, y_true)
    index_d['nd'] = nd(y_pred, y_true)
    index_d['smape'] = SMAPE(y_pred, y_true)
    index_d['mae'] = mean_absolute_error(y_pred, y_true)
    return index_d

def get_mean_index(index_list, key):
    return np.mean([index[key] for index in index_list])
    
def get_mean_index_dict(index_list):
    return { key:get_mean_index(index_list, key) for key in index_list[0].keys() }
