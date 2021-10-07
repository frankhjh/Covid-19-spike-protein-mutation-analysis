import pandas as pd
from data_process import data2df,create_mapping_dict,letter2idx
from data_prepare import prepare_data_loader
from train import train_vae
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from model.vae.encoder.lstm_encoder import Gaussian_LSTM_encoder1,Gaussian_LSTM_encoder2
from model.vae.encoder.mlp_encoder import MLP_encoder
from model.vae.decoder.lstm_decoder import LSTM_decoder
from model.vae.decoder.mlp_decoder import MLP_decoder
from model.vae.VAE import *
from model.vae.model_config import model_parameters
import argparse

parser=argparse.ArgumentParser(description='Parameters for training')
parser.add_argument('--model_type',type=str)
parser.add_argument('--epochs',type=int,default=20)
parser.add_argument('--lr',type=float,default=1e-3)
parser.add_argument('--device',type=str,default='cpu')
parser.add_argument('--period',type=int)
args=parser.parse_args()

model_type=args.model_type
epochs=args.epochs
lr=args.lr
device=args.device
period=args.period

# data prepare
data_dir='./data/output.fasta'
df=data2df(data_dir)
print('>>DataFrame loaded.')

num_aa_types,mapping_dict=create_mapping_dict(df)
print('>>mapping_dict created.')

# create sub-df within given period
curr_df=letter2idx(df,mapping_dict,period)
next_df=letter2idx(df,mapping_dict,period+1)
print('>>sub DataFrames created.')

# create data loaders
if model_type in ['LSTM-LSTM','LSTM-MLP']:
    train_dataloader,val_dataloader=prepare_data_loader(curr_df,train=True,binary=False)
    test_dataloader=prepare_data_loader(next_df,train=False,binary=False)
elif model_type in ['MLP-MLP']:
    train_dataloader,val_dataloader=prepare_data_loader(curr_df,train=True,binary=True)
    test_dataloader=prepare_data_loader(next_df,train=False, binary=True)
print('>>dataloader prepared.')

# train
model=train_vae(model_type=model_type,
          model_parameters=model_parameters,
          train_dataloader=train_dataloader,
          val_dataloader=val_dataloader,
          epochs=epochs,
          lr=lr,
          device=device,
          period=period)

# compute reconstruction error
model.load_state_dict(torch.load(f'./tmp/{model_type}/bm{period}.ckpt'))
print('>>model loaded.')

reconstruction_error=0.0
for step,x in tqdm(enumerate(test_dataloader)):
    error=model.compute_reconstruct_error(x)
    reconstruction_error+=error.item()
reconstruction_error/=(step+1)
print('>>reconstruction error computed.')

with open(f'./tmp/reconstruction_error/{model_type}.txt','w+') as f:
    f.write('{} {}\n'.format(period,reconstruction_error))
f.close()
print('>>Done.')










