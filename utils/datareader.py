import os
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

def data_reader(data_dir):
    output=[]
    
    with open(data_dir,'r') as sp_seqs:
        for line in tqdm(sp_seqs):
            if line.startswith('>'):
                output.append([line.strip()])
            else:
                output[-1].append(line.strip())
    return output



