import numpy as np
import pandas as pd


def dataInfo(set):
    num_ones = [0] * (set.shape[1]
                      )  # [A,B,C,...],A:含有0个"1"的数据条数,B:含有1个"1"的数据条数,...
    for i in range(set.shape[0]):
        n = 0
        for j in range(1, set.shape[1]):
            n += int(set.iloc[i, j])
        num_ones[n] += 1
    for n, num in enumerate(num_ones):
        print("{0}*'1' numbers: {1}".format(n, num))
    return num_ones
