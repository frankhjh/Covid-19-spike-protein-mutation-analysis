import torch
import torch.nn as nn
from ..bn import BN_Layer

class MLP_encoder(nn.Module):
    def __init__(self,dim_z,num_aa_types,seq_len,hidden_units,tau):
        super(MLP_encoder,self).__init__()
        self.dim_z=dim_z
        self.num_aa_types=num_aa_types
        self.seq_len=seq_len
        self.bin_rep_len=seq_len*num_aa_types
        self.hidden_units=hidden_units

        self.encoder_linears=nn.ModuleList()

        self.encoder_linears.append(nn.Linear(self.bin_rep_len,self.hidden_units[0]))
        for i in range(1,len(self.hidden_units)):
            self.encoder_linears.append(nn.Linear(self.hidden_units[i-1],self.hidden_units[i]))
        
        self.encoder_mean=nn.Linear(self.hidden_units[-1],self.dim_z,bias=True)
        self.encoder_logvar=nn.Linear(self.hidden_units[-1],self.dim_z,bias=True)

        self.bn_layer1=BN_Layer(dim_z,tau,mu=True)
        self.bn_layer2=BN_Layer(dim_z,tau,mu=False)
    
    def forward(self,x): # x: (batch_size , seq_len*num_aa_types)
        for hidden_layer in self.encoder_linears:
            x=hidden_layer(x)
            x=torch.tanh(x)
        mean=self.bn_layer1(self.encoder_mean(x)) # mean: (batch_size,dim_z )
        var=torch.exp(self.bn_layer2(self.encoder_logvar(x))) # var: (batch_size,dim_z )
        
        return mean,var