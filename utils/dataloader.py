#!usr/bin/env python
import os
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

def data_reader(data_dir):
    output=[]
    start=0
    
    with open(data_dir,'r') as sp_seqs:
        new_seq=[]
        for line in tqdm(sp_seqs):
            if line.startswith('>'):
                start+=1
                if start==1:
                    new_seq.append(line.strip())
                else:
                    last_seq=new_seq
                    output.append(last_seq)
                    new_seq=[line.strip()]
            else:
                new_seq.append(line.strip())
    #append the last one
    output.append(new_seq)
    return output




