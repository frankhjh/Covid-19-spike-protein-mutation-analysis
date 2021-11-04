import torch
import torch.nn as nn
from ..bn import BN_Layer

class multi_kernel_cnn(nn.Module):
    def __init__(self,embed_dim,vocab_size,feat_size,kernel_sizes,dim_z,tau):
        super(multi_kernel_cnn,self).__init__()
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size
        self.feat_size=feat_size
        self.kernel_sizes=kernel_sizes
        self.dim_z=dim_z
        self.tau=tau

        self.embed_layer=nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim)
        
        self.convs=nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels=self.embed_dim,
                                           out_channels=self.feat_size,
                                           kernel_size=self.kernel_sizes[i],
                                           stride=1),
                                  nn.ReLU(),
                                  nn.AvgPool1d(kernel_size=32)) for i in range(len(self.kernel_sizes))])
        self.bn_layer1=BN_Layer(dim_z,tau,mu=True)
        self.bn_layer2=BN_Layer(dim_z,tau,mu=False)
    
    def forward(self,x):
        embeds=self.embed_layer(x) # (batch_size,seq_len,embed_dim)
        embeds=embeds.permute(0,2,1) # (batch_size,embed_dim,seq_len)

        conv_outs=[conv(embeds) for conv in self.convs] 
        conv_outs_comb=[conv_out.view(conv_out.size(0),-1) for conv_out in conv_outs] # combine the last 2 dims

        tmp_out=torch.cat(conv_outs_comb,dim=1)
        feat_size=tmp_out.size(1)
        out=nn.Linear(feat_size,2*self.dim_z)(tmp_out)
        mean,logvar=out.chunk(2,-1)

        mean=self.bn_layer1(mean) # (batch_size,dim_z)
        var=torch.exp(self.bn_layer2(logvar)) # (batch_size,dim_z)

        return mean,var

class multi_kernel_cnn2(nn.Module):
    def __init__(self,embed_dim,vocab_size,seq_len,feat_size,kernel_sizes,dim_z,tau):
        super(multi_kernel_cnn2,self).__init__()
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size
        self.seq_len=seq_len
        self.feat_size=feat_size
        self.kernel_sizes=kernel_sizes
        self.dim_z=dim_z
        self.tau=tau

        self.embed_layer=nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim)
        
        self.convs=nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels=self.embed_dim,
                                           out_channels=self.feat_size,
                                           kernel_size=self.kernel_sizes[i],
                                           stride=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=self.seq_len-self.kernel_sizes[i]+1)) for i in range(len(self.kernel_sizes))])
        self.linear_mean=nn.Linear(len(self.kernel_sizes)*self.feat_size,dim_z)
        self.linear_logvar=nn.Linear(len(self.kernel_sizes)*self.feat_size,dim_z)
        self.bn_layer1=BN_Layer(dim_z,tau,mu=True)
        self.bn_layer2=BN_Layer(dim_z,tau,mu=False)

    def forward(self,x): # x:(batch_size,seq_len)
        embeds=self.embed_layer(x) # (batch_size,seq_len,embed_dim)
        embeds=embeds.permute(0,2,1) # (batch_size,embed_dim,seq_len)

        conv_outs=[conv(embeds) for conv in self.convs]  # each item : (batch_size,feat_size,1)

        comb_out=torch.cat(conv_outs,dim=1).squeeze(-1) # (batch_size, # of kernel * feat_size)

        mean=self.bn_layer1(self.linear_mean(comb_out))
        var=torch.exp(self.bn_layer2(self.linear_logvar(comb_out)))

        return mean,var
        







