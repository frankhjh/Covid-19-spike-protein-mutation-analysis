#!/usr/bin/env python
import torch
import torch.nn as nn

# reference paper:https://arxiv.org/abs/2004.12585
# class BN_Layer(nn.Module):
#     def __init__(self,dim_z,tau):
#         super(BN_Layer,self).__init__()
#         self.dim_z=dim_z
#         self.bn=nn.BatchNorm1d(dim_z,affine=False)
        
#         self.tau=tau # tau : float in range (0,1)
#         self.theta=torch.randn(1,requires_grad=True)
#         self.gamma1=torch.sqrt(tau+(1-tau)*torch.sigmoid(self.theta)) # for mu
#         self.gamma2=torch.sqrt((1-tau)*torch.sigmoid(-1*self.theta)) # for var
        
#     def forward(self,x,mu=True): # x:(batch_size,dim_z)
#         x=self.bn(x)
#         if mu:
#             x=self.gamma1*x
#         else:
#             x=self.gamma2*x
#         return x

class BN_Layer(nn.Module):
    def __init__(self,dim_z,tau,mu=True):
        super(BN_Layer,self).__init__()
        self.dim_z=dim_z

        
        self.tau=torch.tensor(tau) # tau : float in range (0,1)
        self.theta=torch.tensor(0.5)
       
        self.gamma1=torch.sqrt(self.tau+(1-self.tau)*torch.sigmoid(self.theta)) # for mu
        self.gamma2=torch.sqrt((1-self.tau)*torch.sigmoid((-1)*self.theta)) # for var

        self.bn=nn.BatchNorm1d(dim_z)
        self.bn.bias.requires_grad=False
        self.bn.weight.requires_grad=False

        if mu:
          self.bn.weight.fill_(self.gamma1)
        else:
          self.bn.weight.fill_(self.gamma2)
        
    def forward(self,x): # x:(batch_size,dim_z)
        x=self.bn(x)
        return x




    