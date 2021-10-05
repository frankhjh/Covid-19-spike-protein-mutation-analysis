from torch.utils.data import Dataset

class protein_seq_dataset(Dataset):
    def __init__(self,seq):
        super(protein_seq_dataset,self).__init__()
        self.seq=seq
    
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self,idx):
        return self.seq[idx]
        