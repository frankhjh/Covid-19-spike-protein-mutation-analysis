#!/usr/bin/env python
from utils.dataloader import data_reader

def data2df(data_dir):
    #read data
    output=data_reader(data_dir)
    #build dictionary for store data
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
    
    return seq_df

def sub_df(gidx,num_groups=20):





# encoding the protein sequence of the specific group
def letter2idx(df,gidx): 
    # gidx:group index
    letter_set=set()
    
    unique_seqs=df.sequence.unique()
    for i in range(len(unique_seqs)):
        seq_li=[unique_seqs[i][j] for j in range(len(unique_seqs[i]))]
        letter_set.update(seq_li)
    
    mapping_dict={}
    for l in letter_set:
        mapping_dict[l]=len(mapping_dict)
    
    df['idx_sequence']=df.sequence.apply(lambda x:[x[i] for i in range(len(x))]).\
        apply(lambda x:[mapping_dict.get(i) for i in x])
    
    return df




