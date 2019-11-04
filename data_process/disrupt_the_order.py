# 打乱重新分配训练集和验证集
import os

import pandas as pd

dataPath = r"./data/"


def disruptOrder(set, dataPath=dataPath):
    # print(set.head())
    # print(set.tail())
    a = int(set.shape[0] * 0.75)
    train = set.iloc[:a].reset_index(drop=True)
    valid = set.iloc[a:].reset_index(drop=True)
    # print(train.tail())
    # print(valid.tail())
    train.to_csv(dataPath + 'train.csv', index=False, header=True)
    valid.to_csv(dataPath + 'valid.csv', index=False, header=True)
