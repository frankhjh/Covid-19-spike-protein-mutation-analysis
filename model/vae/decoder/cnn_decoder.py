import torch
import torch.nn as nn

class CNN_decoder(nn.Module):
    def __init__(self,dim_z,kernel_sizes,seq_len,num_aa_types):
        super(CNN_decoder,self).__init__()
        self.dim_z=dim_z
        self.kernel_sizes=kernel_sizes
        self.seq_len=seq_len
        self.num_aa_types=num_aa_types

        self.trans_convs=nn.ModuleList([nn.Sequential(nn.ConvTranspose1d(in_channels=1,
                                                                         out_channels=self.num_aa_types,
                                                                         kernel_size=self.kernel_sizes[i],
                                                                         stride=1),
                                                      nn.ReLU()) for i in range(len(self.kernel_sizes))])
    


    def forward(self,z): # z :(batch_size,dim_z)
        z=z.unsqueeze(1) # z :(batch_size,1,dim_z)

        trans_convs_outs=[trans_conv(z) for trans_conv in self.trans_convs]

        comb_outs=torch.cat(trans_convs_outs,dim=-1)

        out=nn.Linear(comb_outs.size(-1),self.seq_len)(comb_outs) # (batch_size,num_aa_types,seq_len)

        out=out.permute(0,2,1) # (batch_size,seq_len,num_aa_types)

        out=out.reshape(-1,self.num_aa_types) # (batch_size*seq_len,num_aa_types)

        return out