import csv
import pandas as pd
import jsonlines
import numpy as np
from tqdm import tqdm
df = pd.read_csv('ROCStories_winter2017.csv',sep=',')
print(df.tail())
file = jsonlines.open("train.jsonl","w")
for index, row in tqdm(df.iterrows()):
    if index == 42131:
        file.close()
        file = jsonlines.open("test.jsonl","w")
    elif index == 47397:
        file.close()
        file = jsonlines.open("valid.jsonl","w")
    perm = np.random.permutation(5).tolist()
    tmp_dict = {}
    tmp_dict["orig_sents"] = []
    tmp_dict["shuf_sents"] = [' ',' ',' ',' ',' ']
    for i in range(5):
        tmp_dict["orig_sents"].append(str(perm[i]))
        tmp_dict["shuf_sents"][i] = row.iloc[perm.index(i) + 2]
    jsonlines.Writer.write(file,tmp_dict)
file.close()