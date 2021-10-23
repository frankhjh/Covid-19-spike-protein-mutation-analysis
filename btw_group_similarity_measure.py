import pandas as pd
import numpy as np
import json
import argparse

parser=argparse.ArgumentParser(description='months')
parser.add_argument('--year',type=int)
parser.add_argument('--month',type=int)
parser.add_argument('--method',type=str)
args=parser.parse_args()

def prob_each_aa(li,aa_idxs):
    probs=[]
    for aa_idx in aa_idxs:
        probs.append(li.count(aa_idx)/len(li))
    return probs

def kl_divergence(p1,p2):
    assert len(p1)==len(p2)
    kl=0.0
    for i in range(len(p1)):
        kl+=p1[i]*np.log2(p1[i]/p2[i])
    return kl

def seq_same_ratio(seq1,seq2):
    assert len(seq1)==len(seq2)
    return np.mean([seq1[i]==seq2[i] for i in range(len(seq1))])

# use kl divergence to measure group similarity
def cal_group_kl_divergence(df1,df2,sample_size=5000,seq_len=1274,aa_idxs): # aa_idxs:[0,1,2,..,21]
    df1=df1.sample(sample_size).reset_index(drop=True)
    seqs1=df1.idx_sequence.tolist()
    slots_dist1=[]
    for j in range(seq_len):
        slots_dist1.append([seq1[i][j] for i in range(len(seq1))])
    slots_prob1=[]
    for slot_dist1 in slots_dist1:
        slots_prob1.append(prob_each_aa(slot_dist1,aa_idxs))
    
    df2=df2.sample(sample_size).reset_index(drop=True)
    seqs2=df2.idx_sequence.tolist()
    slots_dist2=[]
    for j in range(seq_len):
        slots_dist2.append([seq2[i][j] for i in range(len(seq2))])
    slots_prob2=[]
    for slot_dist2 in slots_dist2:
        slots_prob2.append(prob_each_aa(slot_dist2,aa_idxs))
    
    # cal KL divergence between slots_prob1 & slots_prob2
    avg_kl=0.0
    for p1,p2 in zip(slots_prob1,slots_prob2):
        avg_kl+=kl_divergence(p1,p2)/seq_len
    
    return avg_kl

# use average seq same ratio for 2 groups (a little bit time-consuming)
def cal_group_similarity(df1,df2,sample_size):
    df1,df2=df1.sample(sample_size).reset_index(drop=True),df2.sample(sample_size).reset_index(drop=True)
    seqs1,seqs2=df1.idx_sequence.tolist(),df2.idx_sequence.tolist()
    
    sum_sim=0.0
    for seq1 in tqdm(seqs1):
        for seq2 in seqs2:
            sum_sim+=seq_same_ratio(seq1,seq2)
    avg_sim=sum_sim/(len(seqs1)*len(seqs2))
    return avg_sim


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

    with open('./utils/mapping.json','r') as f:
        mapping_dict=json.load(f)
    aa_idxs=list(mapping_dict.values())
    
    # use either kl_divergence or avg_similarity
    if method=='kl_divergence':
        kl_divergence=cal_group_kl_divergence(df1,df2,sample_size=5000,seq_len=1274,aa_idxs=aa_idxs)
        with open(f'./tmp/similarity_measure/{method}.txt','a') as f:
            f.write(f'{}_{} {}\n'.format(year,month,kl_divergence))
    elif method=='similarity':
        similarity=cal_group_similarity(df1,df2,sample_size=1000)
        with open(f'./tmp/similarity_measure/{method}.txt','a') as f:
            f.write(f'{}_{} {}\n'.format(year,month,similarity))
            






