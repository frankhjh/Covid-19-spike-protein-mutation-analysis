import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict
from utils.datareader import data_reader

def data2df(data_dir):
    # read data
    output=data_reader(data_dir)
    # build dictionary for storing data
    res_dict=defaultdict(list)
    
    for i in tqdm(range(len(output))):
        res_dict['date'].append(output[i][0].split('|')[2])
        res_dict['country'].append(output[i][0].split('|')[-1])
        res_dict['sequence'].append(''.join(output[i][1:]))
    seq_df=pd.DataFrame(res_dict)

    # sort by date,country
    seq_df.sort_values(['date','country'],inplace=True)

    # focus on the sequences after reference sequence.
    seq_df=seq_df[seq_df.date>='2019-12-30'].reset_index(drop=True)

    # drop the duplicate sequences
    # seq_df.drop_duplicates(subset=['sequence'],keep='first',inplace=True)
    # seq_df.reset_index(drop=True,inplace=True)

    # drop the data with unnormal date ,i.e. month=00 or day=00
    unnormal_idx=[]
    for index,row in tqdm(seq_df.iterrows()):
        if '00' in row['date']:
            unnormal_idx.append(index)
    seq_df.drop(unnormal_idx,axis=0,inplace=True)
    seq_df.reset_index(drop=True,inplace=True)
    seq_df.date=pd.to_datetime(seq_df.date,format="%Y-%m-%d")
    
    return seq_df

def group_split(df,gidx,num_groups=10):
    # gidx from 1 to num_groups
    time_diff=df.date.max()-df.date.min()
    span=time_diff.days//num_groups

    start=df.date.min()+pd.Timedelta(f'{span * (gidx-1)} days')
    if gidx!=num_groups:
        end=df.date.min()+pd.Timedelta(f'{span * gidx} days')
        return df[(df.date>=start) & (df.date<end)].reset_index(drop=True)
    else:
        end=df.date.max()
        return df[(df.date>=start) & (df.date<=end)].reset_index(drop=True)

def group_split_month(df,year,month):
    sub_df=df[df.date.apply(lambda x:(x.year,x.month))==(year,month)]
    return sub_df.reset_index(drop=True)

def create_mapping_dict(df):
    letter_set=set()
    
    unique_df=df.drop_duplicates(subset=['sequence'],keep='first')
    unique_df.reset_index(drop=True,inplace=True)
    
    seqs=unique_df.sequence.tolist() # if directly use df.sequence for iteration,very slow!
    for i in tqdm(range(len(seqs))):
        seq_li=[seqs[i][j] for j in range(len(seqs[i]))]
        letter_set.update(seq_li)
    
    mapping_dict={}
    for l in sorted(letter_set):
        mapping_dict[l]=len(mapping_dict)
    
    return len(letter_set),mapping_dict
    
# encoding the seq within speific group
def letter2idx(df,mapping_dict):
    df['idx_sequence']=df.sequence.apply(lambda x:[x[i] for i in range(len(x))][:1273]).\
        apply(lambda x:[mapping_dict.get(i) for i in x])
    return df

if __name__=='__main__':
    data_dir='./data/output.fasta'
    df=data2df(data_dir)

    num_aa_types,mapping_dict=create_mapping_dict(df)
    with open('./utils/mapping.json','w') as f:
        json.dump(mapping_dict,f)

    jan_2020=group_split_month(df,2020,1)
    feb_2020=group_split_month(df,2020,2)
    mar_2020=group_split_month(df,2020,3)
    apr_2020=group_split_month(df,2020,4)
    may_2020=group_split_month(df,2020,5)
    jun_2020=group_split_month(df,2020,6)
    jul_2020=group_split_month(df,2020,7)
    aug_2020=group_split_month(df,2020,8)
    sep_2020=group_split_month(df,2020,9)
    oct_2020=group_split_month(df,2020,10)
    nov_2020=group_split_month(df,2020,11)
    dec_2020=group_split_month(df,2020,12)
    jan_2021=group_split_month(df,2021,1)
    feb_2021=group_split_month(df,2021,2)
    mar_2021=group_split_month(df,2021,3)
    apr_2021=group_split_month(df,2021,4)
    may_2021=group_split_month(df,2021,5)
    jun_2021=group_split_month(df,2021,6)
    jul_2021=group_split_month(df,2021,7)

    data_store=pd.HDFStore('./data/month_split_data.h5')
    data_store['jan2020']=jan_2020
    data_store['feb2020']=feb_2020
    data_store['mar2020']=mar_2020
    data_store['apr2020']=apr_2020
    data_store['may2020']=may_2020
    data_store['jun2020']=jun_2020
    data_store['jul2020']=jul_2020
    data_store['aug2020']=aug_2020
    data_store['sep2020']=sep_2020
    data_store['oct2020']=oct_2020
    data_store['nov2020']=nov_2020
    data_store['dec2020']=dec_2020
    data_store['jan2021']=jan_2021
    data_store['feb2021']=feb_2021
    data_store['mar2021']=mar_2021
    data_store['apr2021']=apr_2021
    data_store['may2021']=may_2021
    data_store['jun2021']=jun_2021
    data_store['jul2021']=jul_2021

    data_store.close()



















