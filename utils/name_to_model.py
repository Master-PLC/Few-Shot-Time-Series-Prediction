#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :name_to_model.py
@Description  :
@Date         :2021/11/08 15:03:54
@Author       :Arctic Little Pig
@Version      :1.0
'''

import xgboost
from models.bhtarima import BHTARIMA
from models.xgboost import XGBoost

NAME_TO_MODEL_MAP = {
    "xgboost": XGBoost,
    "bht_arima": BHTARIMA
}


def name2model(model_name: str) -> object:
    model_name = model_name.lower()

    return NAME_TO_MODEL_MAP[model_name]
