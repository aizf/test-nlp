import pandas as pd
import os
dataPath=r"./data"
dataList = os.listdir(dataPath)
set = pd.read_csv(r"./data/1.csv")
print(set.head())