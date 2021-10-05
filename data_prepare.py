from torch.utils.data import DataLoader
from utils.dataset import protein_seq_dataset
import torch
import pandas as pd

def idx2binary_rep(idx_seq): # idx_seq like [1,3,4,2,5,2,1,4,5]
    out=[]
    for idx in idx_seq:
        out+=[1 if i==idx else 0 for i in range(22)] # num_aa_types=22
    return out


def prepare_data_loader(df,train=True,binary=True):
   if train:
        df=df.sample(frac=1).reset_index(drop=True)
        size=df.shape[0]
        if binary:
            df['binary_rep_sequence']=df.idx_sequence.apply(idx2binary_rep)
            train_seq_li=df.binary_rep_sequence.tolist()[:int(0.9*size)]
            val_seq_li=df.binary_rep_sequence.tolist()[int(0.9*size):]
        else:
            train_seq_li=df.idx_sequence.tolist()[:int(0.9*size)]
            val_seq_li=df.idx_sequence.tolist()[int(0.9*size):]

        train_tensor=torch.Tensor(train_seq_li)
        val_tensor=torch.Tensor(val_seq_li)

        train_set=protein_seq_dataset(train_tensor)
        val_set=protein_seq_dataset(val_tensor)

        train_dataloader=DataLoader(train_set,batch_size=32,shuffle=True)
        val_dataloader=DataLoader(val_set,batch_size=32,shuffle=False)

        return train_dataloader,val_dataloader
    else:
        if binary:
            df['binary_rep_sequence']=df.idx_sequence.apply(idx2binary_rep)
            test_seq_li=df.binary_rep_sequence.tolist()
        else:
            test_seq_li=df.idx_sequence.tolist()
        test_tensor=torch.Tensor(test_seq_li)
        test_set=protein_seq_dataset(test_tensor)
        test_dataloader=DataLoader(test_set)
        return test_dataloader
 


        
        