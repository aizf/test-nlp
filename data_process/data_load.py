import numpy as np
import pandas as pd

dataPath = r"./data/"
dataList = ["train.csv", "valid.csv"]


def loadData(dataPath=dataPath, dataList=dataList):
    print("*****")
    for data in dataList:
        print("Loaded {0} from {1}".format(data, dataPath))
    print("*****")
    dataList = [pd.read_csv(dataPath + data) for data in dataList]
    set = pd.concat(dataList, ignore_index=True)
    return set
