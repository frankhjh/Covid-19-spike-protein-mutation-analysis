#!/usr/bin/env python

# encoding the protein sequence
def letter2idx(df): 
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

def grouping(df,num): # num: number of group 
    pass



