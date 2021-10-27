import torch
import torch.nn as nn
from ..bn import BN_Layer

# single z
class Gaussian_LSTM_encoder1(nn.Module):
    def __init__(self,embed_dim,vocab_size,hid_size,dim_z,tau):
        super(Gaussian_LSTM_encoder1,self).__init__()
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size
        self.hid_size=hid_size
        self.dim_z=dim_z
        
        self.embed_layer=nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim)
        
        self.lstm=nn.LSTM(input_size=embed_dim,
                          hidden_size=hid_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        
        self.linear2mean=nn.Linear(2*hid_size,dim_z)
        self.linear2logvar=nn.Linear(2*hid_size,dim_z)
        self.bn_layer1=BN_Layer(dim_z,tau,mu=True)
        self.bn_layer2=BN_Layer(dim_z,tau,mu=False)
    
    def forward(self,x):
        embeddings=self.embed_layer(x) # (batch_size,seq_len,embed_dim)
        _,(last_hid,last_cell)=self.lstm(embeddings) # last_hid:(2*1,batch_size,hid_size)

        last_hid=last_hid.permute(1,0,2)
        bid_last_hid=last_hid.reshape(last_hid.size(0),-1) # tmp:(batch_size,2*hid_size)
        
        
        mean=self.linear2mean(bid_last_hid)
        logvar=self.linear2logvar(bid_last_hid) # mean/logvar:(batch_size,dim_z)

        mean=self.bn_layer1(mean) # (batch_size,dim_z)
        var=torch.exp(self.bn_layer2(logvar)) # (batch_size,dim_z)
        return mean,var

# multi z
class Gaussian_LSTM_encoder2(nn.Module):
    def __init__(self,embed_dim,vocab_size,hid_size,dim_z,tau):
        super(Gaussian_LSTM_encoder2,self).__init__()
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size
        self.hid_size=hid_size
        self.dim_z=dim_z
        self.embed_layer=nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim)
        
        self.lstm=nn.LSTM(input_size=embed_dim,
                          hidden_size=hid_size,
                          num_layers=1,
                          batch_first=True)
        self.linear2mean=nn.Linear(hid_size,dim_z,bias=False)
        self.linear2logvar=nn.Linear(hid_size,dim_z,bias=False)
        self.bn_layer1=BN_Layer(dim_z,tau,mu=True)
        self.bn_layer2=BN_Layer(dim_z,tau,mu=False)
    
    def forward(self,x):
        embeddings=self.embed_layer(x) # (batch_size,seq_len,embed_dim)
        lstm_out,(_,_)=self.lstm(embeddings) # lstm_out :(batch_size,seq_len,hid_size)

        means=self.linear2mean(lstm_out) # means :(batch_size,seq_len,dim_z)
        logvars=self.linear2logvar(lstm_out) # logvars :(batch_size,seq_len,dim_z)

        means=self.bn_layer1(means.permute(0,2,1)) # means :(batch_size,dim_z,seq_len)
        var_s=torch.exp(self.bn_layer2(logvars.permute(0,2,1))) # vars :(batch_size,dim_z,seq_len)

        return means,var_s
        










        
