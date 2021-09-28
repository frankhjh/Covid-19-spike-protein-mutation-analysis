#!/usr/bin/env python
import torch
import torch.nn as nn
from ..batch_normalization import BN_Layer

class Gaussian_LSTM_encoder(nn.Module):
    def __init__(self,embed_dim,vocab_size,hid_size,dim_z,tau):
        super(Gaussian_LSTM_encoder,self).__init__()
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size
        self.hid_size=hid_size
        self.dim_z=dim_z
        
        self.embed_layer=nn.Embedding(num_embddings=vocab_size,
                                      embedding_dim=embed_dim)
        
        self.lstm=nn.LSTM(input_size=embed_dim,
                          hidden_size=hid_size,
                          num_layers=1,
                          batch_first=True)
        
        self.linear=nn.Linear(hid_size,2*dim_z,bias=False)
        self.bn_layer=BN_Layer(dim_z,tau)
    
    def forward(self,x):
        embeddings=self.embed_layer(x) # (batch_size,seq_len,embed_dim)
        _,(last_hid,last_cell)=self.lstm(embeddings) # last_hid:(1,batch_size,hid_size)

        mean,logvar=self.linear(last_hid).chunk(2,-1) # mean/logvar:(1,batch_size,dim_z)

        mean=self.bn_layer(mean.squeeze(0)) # (batch_size,dim_z)
        var=self.bn_layer(torch.exp(logvar.squeeze(0)),mu=False) # (batch_size,dim_z)
        return mean,var








        
