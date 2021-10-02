#!/usr/bin/env python
import torch
import torch.nn as nn
from .encoder.lstm_encoder import Gaussian_LSTM_encoder1,Gaussian_LSTM_encoder2
from .encoder.mlp_encoder import MLP_encoder
from .decoder.lstm_decoder import LSTM_decoder
from .decoder.mlp_decoder import MLP_decoder

class vae_gaussian_base(nn.Module):
    def __init__(self,encoder,decoder):
        super(vae_gaussian_base,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
    
    def encode(self,x):
        return self.encoder(x)
    
    def decode(self,x):
        return self.decoder(x)
    
    def reparameterize(self,mean,var):
        # sample z from q(z|x)
        std=torch.sqrt(var)
        eps=torch.randn_like(mean) 
        return mean+eps*std
    
    def compute_reconstruct_error(self,x): 
        # compute the log p(x|z)
        raise NotImplementedError

    def compute_elbo(self,x):
        # compute the total loss 
        raise NotImplementedError

    def forward(self,x):
        raise NotImplementedError


class vae_gaussian_mlp(vae_gaussian_base):
    def __init__(self,encoder,decoder):
        super(vae_gaussian_mlp,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
    
    def compute_reconstruct_error(self,x): # x:(batch_size,seq_len*num_aa_types)
        mean,var=self.encode(x) # mean:(batch_size,dim_z) / var:(batch_size,dim_z)
        z=self.reparameterize(mean,var) # z:(batch_size,dim_z)
        out=self.decode(z) # out:(batch_size,seq_len*num_aa_types) 

        # use cross entropy as loss function
        loss=torch.sum(x*out,-1) # loss:(batch_size,)
        return torch.mean(loss)
    
    def compute_elbo(self,x): # x:(batch_size,seq_len*num_aa_types)
        mean,var=self.encode(x) 
        # KL divergence
        kl_divergence=torch.sum(0.5*(-torch.log(var)+mean**2+var-1),-1) # (batch,)
        
        z=self.reparameterize(mean,var)
        out=self.decode(z)

        # reconstruction error
        loss=torch.sum(x*out,-1) # (batch,)
        elbo=loss-kl_divergence # (batch,)
        return torch.sum(elbo)
    
    def forward(self,x):
        mean,var=self.encode(x)
        z=self.reparameterize(mean,var)
        out=self.decode(z)
        return mean,var,out

class vae_gaussian_lstm(vae_gaussian_base):
    def __init__(self,encoder,decoder):
        super(vae_gaussian_lstm,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
    
    def compute_reconstruct_error(self,x): # x:(batch_size,seq_len)
        means,var_s=self.encode(x) 

        z=self.reparameterize(means,var_s) # z:(batch_size,dim_z,seq_len)
        out=self.decode(z.permute(0,2,1)) # out:(batch_size*seq_len,num_aa_types)

        target=x.view(-1) # target:(batch_size*seq_len,)
        
        cross_entropy_loss=nn.CrossEntropyLoss(reduction='sum')
        loss=cross_entropy_loss(out,target)
        return loss
    
    def compute_elbo(self,x):
        means,var_s=self.encode(x)
        # KL divergence
        kl_divergence=torch.mean(torch.sum(0.5*(-torch.log(var_s)+means**2+var_s-1),-1),-1) # (batch,)
        kl_divergence=torch.sum(kl_divergence)

        z=self.reparameterize(means,var_s) # z:(batch_size,dim_z,seq_len)
        out=self.decode(z.permute(0,2,1)) # out:(batch_size*seq_len,num_aa_types)

        target=x.view(-1) # target:(batch_size*seq_len,)
        
        cross_entropy_loss=nn.CrossEntropyLoss(reduction='sum')
        loss=cross_entropy_loss(out,target)
        elbo=loss-kl_divergence
        return elbo
    
    def forward(self,x):
        means,var_s=self.encode(x)
        z=self.reparameterize(means,var_s)
        out=self.decode(z.permute(0,2,1))

        return out,means,var_s

















