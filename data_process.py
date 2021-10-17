import pandas as pd
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
    seq_df.drop_duplicates(subset=['sequence'],keep='first',inplace=True)
    seq_df.reset_index(drop=True,inplace=True)

    # drop the data with unnormal date ,i.e. month=00 or day=00
    unnormal_idx=[]
    for i in range(seq_df.shape[0]):
        if '00' in seq_df.date[i]:
            unnormal_idx.append(i)
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
    
    seqs=df.sequence.tolist() # if directly use df.sequence for iteration,very slow!
    for i in tqdm(range(len(seqs))):
        seq_li=[seqs[i][j] for j in range(len(seqs[i]))]
        letter_set.update(seq_li)
    
    mapping_dict={}
    for l in letter_set:
        mapping_dict[l]=len(mapping_dict)
    
    return len(letter_set),mapping_dict
    
# encoding the seq within speific group
def letter2idx(df,mapping_dict,year,month): # gidx:group index
    sub_df=group_split_month(df,year,month)
    sub_df['idx_sequence']=sub_df.sequence.apply(lambda x:[x[i] for i in range(len(x))]).\
        apply(lambda x:[mapping_dict.get(i) for i in x])
    
    return sub_df




