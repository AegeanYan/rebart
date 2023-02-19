from itertools import  permutations
import nltk
import jsonlines
import pandas as pd
import json
import torch
import tqdm
from tqdm import tqdm
import numpy as np

df = pd.read_json('../Plansum/data/amazon/train.plan.json')
file = jsonlines.open("./data/amazon_full_5/train.jsonl","w")
for i in tqdm(range(90000)):
    if i == 72000:
        file.close()
        file = jsonlines.open("./data/amazon_full_5/dev.jsonl","w")
    elif i == 81000:
        file.close()
        file = jsonlines.open("./data/amazon_full_5/test.jsonl","w")
    
    for j in range(9):
        if j == 8:
            tokens = nltk.sent_tokenize(df.iloc[i,1])
        else:
            tokens = nltk.sent_tokenize(df.iloc[i,0][j][0])
        lenth = len(tokens)
        if(lenth != 5):continue
        perm = np.random.permutation(5).tolist()
        tmp_dict = {}
        tmp_dict["orig_sents"] = []
        tmp_dict["shuf_sents"] = [' ',' ',' ',' ',' ']
        for i in range(5):
            tmp_dict["orig_sents"].append(str(perm[i]))
            tmp_dict["shuf_sents"][i] = tokens[perm.index(i)]
        jsonlines.Writer.write(file,tmp_dict)
    
file.close()