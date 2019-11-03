import pandas as pd
import os
dataPath = r"./data/"
# __dataList = os.listdir(dataPath)  # data file names
__dataList = ["train.csv","valid.csv"]
dataList = [pd.read_csv(dataPath + data) for data in __dataList]
set = pd.concat(dataList, ignore_index=True)
# set=set.sample(frac=1).reset_index(drop=True)
print(set.head())
print(set.tail())
a = int(set.shape[0]*0.75)
train = set.iloc[:a].reset_index(drop=True)
valid = set.iloc[a:].reset_index(drop=True)
print(train.tail())
print(valid.tail())
train.to_csv(dataPath + 'train.csv', index=False, header=True)
valid.to_csv(dataPath + 'valid.csv', index=False, header=True)
