from torch.utils.data import DataLoader
from utils.dataset import protein_seq_dataset
from utils.site_entropy import site_entropy_in_group
import torch
import json
import pandas as pd


def idx2binary_rep(idx_seq): # idx_seq like [1,3,4,2,5,2,1,4,5]
    out=[]
    for idx in idx_seq:
        out+=[1 if i==idx else 0 for i in range(22)] # num_aa_types=22
    return out

#for non-unique df
def prepare_data_loader(df,fix_sizes=[9000,1000],train=True,binary=True):
    if train:
        unique_df=df.drop_duplicates(subset=['sequence'],keep='first')
        unique_df.reset_index(drop=True,inplace=True)

        unique_df_train=unique_df.iloc[:int(len(unique_df)*0.9),:]
        unique_df_val=unique_df.iloc[int(len(unique_df)*0.9):,:].reset_index(drop=True)
    
        df_train=df[df.sequence.isin(pd.Series(unique_df_train.sequence.tolist()))].reset_index(drop=True)
        df_val=df[df.sequence.isin(pd.Series(unique_df_val.sequence.tolist()))].reset_index(drop=True)
   
        train_df=unique_df_train.sample(fix_sizes[0]).reset_index(drop=True) if unique_df_train.shape[0]>=fix_sizes[0] \
            else unique_df_train.append(df_train.sample(fix_sizes[0]-unique_df_train.shape[0],replace=True).reset_index(drop=True)).reset_index(drop=True)
     
        val_df=unique_df_val.sample(fix_sizes[1]).reset_index(drop=True) if unique_df_val.shape[0]>=fix_sizes[1] \
            else unique_df_val.append(df_val.sample(fix_sizes[1]-unique_df_val.shape[0],replace=True).reset_index(drop=True)).reset_index(drop=True)
    
        if binary:
            train_df['binary_rep_sequence']=train_df.idx_sequence.apply(idx2binary_rep)
            train_seq_li=train_df.binary_rep_sequence.tolist()
            val_df['binary_rep_sequence']=val_df.idx_sequence.apply(idx2binary_rep)
            val_seq_li=val_df.binary_rep_sequence.tolist()

            train_tensor=torch.Tensor(train_seq_li)
            val_tensor=torch.Tensor(val_seq_li)
        else:
            train_seq_li=train_df.idx_sequence.tolist()
            val_seq_li=val_df.idx_sequence.tolist()

            train_tensor=torch.tensor(train_seq_li)
            val_tensor=torch.tensor(val_seq_li)

        train_set=protein_seq_dataset(train_tensor)
        val_set=protein_seq_dataset(val_tensor)

        train_dataloader=DataLoader(train_set,batch_size=32,shuffle=True)
        val_dataloader=DataLoader(val_set,batch_size=32,shuffle=False)

        return train_dataloader,val_dataloader
    else:
        # only use unique sequence for testing
        df=df.drop_duplicates(subset=['sequence'],keep='first').reset_index(drop=True)
        size=df.shape[0]
        final_size=32*(size//32)
        df=df.sample(final_size).reset_index(drop=True)
        if binary:
            df['binary_rep_sequence']=df.idx_sequence.apply(idx2binary_rep)
            test_seq_li=df.binary_rep_sequence.tolist()
            test_tensor=torch.Tensor(test_seq_li)
        else:
            test_seq_li=df.idx_sequence.tolist()
            test_tensor=torch.tensor(test_seq_li)
        
        test_set=protein_seq_dataset(test_tensor)
        test_dataloader=DataLoader(test_set,batch_size=32,shuffle=False)
        return test_dataloader

# for unique df
def prepare_data_loader2(df,fix_size=5500,train=True,binary=True):
    if train:
        df_sample=df.sample(fix_size).reset_index(drop=True)
        train_df=df_sample.iloc[:5000,:].copy()
        entropy_index=site_entropy_in_group(train_df,seq_len=1273) # calculate the similarity within group
        
        with open('./utils/overall_site_entropy.json','r') as f:
            overall_site_entropy=json.load(f)[0]
        
        if entropy_index>overall_site_entropy:
            fair_size=int((overall_site_entropy/entropy_index)*5000)
            if fair_size%32==1:
                fair_size-=1
            train_df=train_df.sample(fair_size).reset_index(drop=True)
        
        val_df=df_sample.iloc[5000:,:].reset_index(drop=True)
        if binary:
            train_df['binary_rep_sequence']=train_df.idx_sequence.apply(idx2binary_rep)
            train_seq_li=train_df.binary_rep_sequence.tolist()
            val_df['binary_rep_sequence']=val_df.idx_sequence.apply(idx2binary_rep)
            val_seq_li=val_df.binary_rep_sequence.tolist()

            train_tensor=torch.Tensor(train_seq_li)
            val_tensor=torch.Tensor(val_seq_li)
        else:
            train_seq_li=train_df.idx_sequence.tolist()
            val_seq_li=val_df.idx_sequence.tolist()

            train_tensor=torch.tensor(train_seq_li)
            val_tensor=torch.tensor(val_seq_li)

        train_set=protein_seq_dataset(train_tensor)
        val_set=protein_seq_dataset(val_tensor)

        train_dataloader=DataLoader(train_set,batch_size=32,shuffle=True)
        val_dataloader=DataLoader(val_set,batch_size=32,shuffle=False)

        return entropy_index,train_dataloader,val_dataloader
    else:
        size=df.shape[0]
        final_size=32*(size//32)
        df=df.sample(final_size).reset_index(drop=True)
        if binary:
            df['binary_rep_sequence']=df.idx_sequence.apply(idx2binary_rep)
            test_seq_li=df.binary_rep_sequence.tolist()
            test_tensor=torch.Tensor(test_seq_li)
        else:
            test_seq_li=df.idx_sequence.tolist()
            test_tensor=torch.tensor(test_seq_li)
        
        test_set=protein_seq_dataset(test_tensor)
        test_dataloader=DataLoader(test_set,batch_size=32,shuffle=False)
        return test_dataloader

def prepare_data_loader_region(df,region,fix_size,train=True,binary=True):
    if train:
        df_sample=df.sample(fix_size).reset_index(drop=True)
        train_df=df_sample.iloc[:fix_size*10/11,:].copy()
        entropy_index=site_entropy_in_group(train_df,seq_len=1273) # calculate the similarity within group
        
        with open(f'./utils/region/{region}_overall_site_entropy.json','r') as f:
            region_overall_site_entropy=json.load(f)[0]
        
        if entropy_index>region_overall_site_entropy:
            fair_size=int((region_overall_site_entropy/entropy_index)*train_df.shape[0])
            if fair_size%32==1:
                fair_size-=1
            train_df=train_df.sample(fair_size).reset_index(drop=True)
        
        val_df=df_sample.iloc[fix_size*10/11:,:].reset_index(drop=True)
        if binary:
            train_df['binary_rep_sequence']=train_df.idx_sequence.apply(idx2binary_rep)
            train_seq_li=train_df.binary_rep_sequence.tolist()
            val_df['binary_rep_sequence']=val_df.idx_sequence.apply(idx2binary_rep)
            val_seq_li=val_df.binary_rep_sequence.tolist()

            train_tensor=torch.Tensor(train_seq_li)
            val_tensor=torch.Tensor(val_seq_li)
        else:
            train_seq_li=train_df.idx_sequence.tolist()
            val_seq_li=val_df.idx_sequence.tolist()

            train_tensor=torch.tensor(train_seq_li)
            val_tensor=torch.tensor(val_seq_li)

        train_set=protein_seq_dataset(train_tensor)
        val_set=protein_seq_dataset(val_tensor)

        train_dataloader=DataLoader(train_set,batch_size=32,shuffle=True)
        val_dataloader=DataLoader(val_set,batch_size=32,shuffle=False)

        return entropy_index,train_dataloader,val_dataloader
    else:
        size=df.shape[0]
        final_size=32*(size//32)
        df=df.sample(final_size).reset_index(drop=True)
        if binary:
            df['binary_rep_sequence']=df.idx_sequence.apply(idx2binary_rep)
            test_seq_li=df.binary_rep_sequence.tolist()
            test_tensor=torch.Tensor(test_seq_li)
        else:
            test_seq_li=df.idx_sequence.tolist()
            test_tensor=torch.tensor(test_seq_li)
        
        test_set=protein_seq_dataset(test_tensor)
        test_dataloader=DataLoader(test_set,batch_size=32,shuffle=False)
        return test_dataloader
        
        