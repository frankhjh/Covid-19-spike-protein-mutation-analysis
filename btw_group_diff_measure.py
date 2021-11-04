from tqdm import tqdm
from data_process import letter2idx
import pandas as pd
import numpy as np
import json
import argparse
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

parser=argparse.ArgumentParser(description='months')
parser.add_argument('--year',type=int)
parser.add_argument('--month',type=int)
parser.add_argument('--continent',type=str,default=None)
parser.add_argument('--method',type=str)
args=parser.parse_args()

def prob_each_aa(li,aa_idxs):
    probs=[]
    for aa_idx in aa_idxs:
        probs.append(li.count(aa_idx)/len(li))
    return probs

def kl_divergence(p1,p2,eps=1e-5):
    assert len(p1)==len(p2)
    kl=0.0
    for i in range(len(p1)):
        kl+=p1[i]*np.log2((p1[i]+eps)/(p2[i]+eps))
    return kl

def seq_diff_ratio(seq1,seq2):
    assert len(seq1)==len(seq2)
    return np.mean([seq1[i]!=seq2[i] for i in range(len(seq1))])

# use kl divergence to measure group mutation
def cal_group_kl_divergence(df1,df2,aa_idxs,sample_size=5000,seq_len=1273): # aa_idxs:[0,1,2,..,21]
    if df1.shape[0]>sample_size:
        df1=df1.sample(sample_size).reset_index(drop=True)
    seqs1=df1.idx_sequence.tolist()
    slots_dist1=[]
    for j in tqdm(range(seq_len)):
        slots_dist1.append([seqs1[i][j] for i in range(len(seqs1))])
    slots_prob1=[]
    for slot_dist1 in slots_dist1:
        slots_prob1.append(prob_each_aa(slot_dist1,aa_idxs))
    
    if df2.shape[0]>sample_size:
        df2=df2.sample(sample_size).reset_index(drop=True)
    seqs2=df2.idx_sequence.tolist()
    slots_dist2=[]
    for j in tqdm(range(seq_len)):
        slots_dist2.append([seqs2[i][j] for i in range(len(seqs2))])
    slots_prob2=[]
    for slot_dist2 in slots_dist2:
        slots_prob2.append(prob_each_aa(slot_dist2,aa_idxs))
    
    # cal KL divergence between slots_prob1 & slots_prob2
    avg_kl=0.0
    for p1,p2 in tqdm(zip(slots_prob1,slots_prob2)):
        avg_kl+=kl_divergence(p1,p2)/seq_len
    
    return avg_kl

# use average seq diff ratio for 2 groups (a little bit time-consuming)
def cal_group_diff(df1,df2,sample_size=2000):
    if df1.shape[0]>sample_size:
        df1=df1.sample(sample_size).reset_index(drop=True)
    if df2.shape[0]>sample_size:
        df2=df2.sample(sample_size).reset_index(drop=True)
    
    seqs1,seqs2=df1.idx_sequence.tolist(),df2.idx_sequence.tolist()
    
    sum_diff=0.0
    for seq1 in tqdm(seqs1):
        for seq2 in seqs2:
            sum_diff+=seq_diff_ratio(seq1,seq2)
    avg_diff=sum_diff/(len(seqs1)*len(seqs2))
    return avg_diff


if __name__=='__main__':
    year=args.year
    month=args.month
    method=args.method

    month_name={1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',
        9:'sep',10:'oct',11:'nov',12:'dec'}
    key1=month_name[month]+str(year)
    key2=month_name[month+1]+str(year) if month!=12 else month_name[1]+str(year+1)

    data_store=pd.HDFStore('./data/month_unique_data.h5')
    df1=data_store[key1]
    df2=data_store[key2]
    data_store.close()
    print('>>DataFrames Loaded.')

    with open('./utils/mapping.json','r') as f:
        mapping_dict=json.load(f)
    aa_idxs=list(mapping_dict.values())

    df1=letter2idx(df1,mapping_dict)
    df2=letter2idx(df2,mapping_dict)
    print('>>Sequence Encoding Done.')

    
    # use either kl_divergence or avg_similarity
    if method=='kl_divergence':
        avg_kld=0.0
        for i in range(20):
            kld=cal_group_kl_divergence(df1,df2,aa_idxs=aa_idxs,sample_size=5000,seq_len=1273)
            avg_kld+=kld/20

        with open(f'./tmp/diff_measure/{method}.txt','a') as f:
            f.write('{}_{} {}\n'.format(year,month,avg_kld))
    
    elif method=='diff_ratio':
        avg_diff_ratio=0.0
        for i in range(10):
            diff_ratio=cal_group_diff(df1,df2,sample_size=1000)
            avg_diff_ratio+=diff_ratio/10
        
        with open(f'./tmp/diff_measure/{method}.txt','a') as f:
            f.write('{}_{} {}\n'.format(year,month,avg_diff_ratio))
            






