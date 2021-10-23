import numpy as np

def slot_entropy(li):
    entropy=0.0
    for i in set(li):
        p=li.count(i)/len(li)
        entropy+=(-1)*p*np.log2(p)
    return entropy

def similarity_in_group(df,seq_len):
    seqs=df.idx_sequence.tolist()
    li_slots=[]
    for j in range(seq_len):
        li_slots.append([seqs[i][j] for i in range(len(seqs))])
    
    # calculate the slot entropy
    avg_entropy=0.0
    for li_slot in li_slots:
        entropy=slot_entropy(li_slot)
        avg_entropy+=entropy/seq_len
    return avg_entropy
