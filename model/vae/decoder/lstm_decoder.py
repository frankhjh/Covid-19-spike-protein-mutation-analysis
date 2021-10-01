#!/usr/bin/env python
import torch
import torch.nn as nn

class LSTM_decoder(nn.Module):
    def __init__(self,dim_z,hid_size,num_aa_types):
        super(LSTM_decoder,self).__init__()
        self.dim_z=dim_z
        self.hid_size=hid_size
        self.num_aa_types=num_aa_types

        self.lstm=nn.LSTM(input_size=dim_z,
                          hidden_size=hid_size,
                          num_layers=1,
                          batch_first=True)
        
        self.linear=nn.Linear(hid_size,num_aa_types)
    
    def forward(self,z): # z: (batch_size,seq_len,dim_z)
        lstm_out,(_,_)=self.lstm(z) # lstm_out: (batch_size,seq_len,hid_size)
        out=self.linear(lstm_out) # out: (batch_size,seq_len,num_aa_types)
        out=out.view(-1,num_aa_types)

        return out

