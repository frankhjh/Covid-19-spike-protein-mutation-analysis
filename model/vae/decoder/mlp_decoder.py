import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_decoder(nn.Module):
    def __init__(self,dim_z,num_aa_types,seq_len,hidden_units):
        super(MLP_decoder,self).__init__()
        self.dim_z=dim_z
        self.num_aa_types=num_aa_types
        self.seq_len=seq_len
        self.bin_rep_len=seq_len*num_aa_types
        
        self.hidden_units=hidden_units
        self.decoder_linears=nn.ModuleList()

        self.decoder_linears.append(nn.Linear(dim_z,self.hidden_units[0]))
        for i in range(len(self.decoder_linears)):
            self.decoder_linears.append(nn.Linear(self.hidden_units[i],self.hidden_units[i+1]))
        
        self.decoder_linears.append(nn.Linear(self.hidden_units[-1],self.bin_rep_len))
    
    def forward(self,z):

        hidden=z
        for i in range(len(self.decoder_linears)-1):
            hidden=self.decoder_linears[i](hidden)
            hidden=torch.tanh(hidden)
        out=self.decoder_linears[-1](hidden)

        out=out.view(-1,self.seq_len,self.num_aa_types)

        log_p=F.log_softmax(out,-1)
        log_p=log_p.view(-1,self.bin_rep_len)

        return log_p



