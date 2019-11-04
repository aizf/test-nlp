import numpy as np
import pandas as pd

from .data_info import dataInfo

OUTPUT_DIR = r"./adjusted_data"


def adjustData(set, num_ones=None, output_dir=OUTPUT_DIR):
    set = set.sample(frac=1)
    if num_ones==None:
        num_ones = dataInfo(set)

    num_0 = num_ones[0]  # 0*'1'
    num_others = np.sum(num_ones[1:])  # 其他*'1'
    num_to_rm = min(int(num_others * 0.6), num_0)
    print("num_0 :", num_0)
    print("num_others :", num_others)
    print("num_to_rm :", num_to_rm)

    for index, row in set.iterrows():
        if num_to_rm <= 0:
            break
        n = 0
        for j in range(1, set.shape[1]):
            n += int(row.iloc[j])
        if n == 0:
            # print(row)
            set.drop([index], inplace=True)
            num_to_rm -= 1

    # print(set.shape)
    set = set.sample(frac=1).reset_index(drop=True)
    a = int(set.shape[0] * 0.8)
    train = set.iloc[:a].reset_index(drop=True)
    valid = set.iloc[a:].reset_index(drop=True)
    # print(train.tail())
    # print(valid.tail())
    train.to_csv(output_dir + '/train.csv', index=False, header=True)
    valid.to_csv(output_dir + '/valid.csv', index=False, header=True)
    return set
