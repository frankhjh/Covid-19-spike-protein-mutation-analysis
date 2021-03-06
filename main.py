import pandas as pd
import json
from data_process import letter2idx
from data_prepare import prepare_data_loader,prepare_data_loader2
from train import train_vae
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from model.vae.encoder.lstm_encoder import Gaussian_LSTM_encoder1,Gaussian_LSTM_encoder2
from model.vae.encoder.mlp_encoder import MLP_encoder
from model.vae.encoder.cnn_encoder import multi_kernel_cnn,multi_kernel_cnn2
from model.vae.decoder.cnn_decoder import CNN_decoder
from model.vae.decoder.lstm_decoder import LSTM_decoder
from model.vae.decoder.mlp_decoder import MLP_decoder
from model.vae.VAE import *
from model.vae.model_config import model_parameters
import argparse
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

parser=argparse.ArgumentParser(description='Parameters for training')
parser.add_argument('--model_type',type=str)
parser.add_argument('--epochs',type=int,default=20)
parser.add_argument('--lr',type=float,default=1e-3)
parser.add_argument('--device',type=str,default='cpu')
parser.add_argument('--year',type=int)
parser.add_argument('--month',type=int)
args=parser.parse_args()

model_type=args.model_type
epochs=args.epochs
lr=args.lr
device=args.device
year=args.year
month=args.month

month_name={1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',
    9:'sep',10:'oct',11:'nov',12:'dec'}

# load data from hdf
data_store=pd.HDFStore('./data/month_unique_data.h5')
key1=month_name[month]+str(year)
key2=month_name[month+1]+str(year) if month!=12 else month_name[1]+str(year+1)

df1=data_store[key1]
df2=data_store[key2]
data_store.close()
print('>>DataFrames Loaded.')

with open('./utils/mapping.json','r') as f:
    mapping_dict=json.load(f)

current_month=letter2idx(df1,mapping_dict)
next_month=letter2idx(df2,mapping_dict)
print('>>Sequence Encoding Done.')

# create data loaders
if model_type in ['LSTM-LSTM','LSTM-MLP','CNN-MLP','CNN-CNN']:
    entropy_index,train_dataloader,val_dataloader=prepare_data_loader2(current_month,fix_size=5500,train=True,binary=False)
    test_dataloader=prepare_data_loader2(next_month,fix_size=5500,train=False,binary=False)
elif model_type in ['MLP-MLP']:
    entropy_index,train_dataloader,val_dataloader=prepare_data_loader2(current_month,fix_size=5500,train=True,binary=True)
    test_dataloader=prepare_data_loader2(next_month,fix_size=5500,train=False, binary=True)
print('>>Dataloader Prepared.')

# train
model=train_vae(model_type=model_type,
          model_parameters=model_parameters,
          train_dataloader=train_dataloader,
          val_dataloader=val_dataloader,
          epochs=epochs,
          lr=lr,
          device=device,
          year=year,
          month=month)

model.load_state_dict(torch.load(f'./tmp/{model_type}/bm_{year}_{month}.ckpt'))
print('>>Model Loaded.')

# get latent z
z_s=[]
if model_type in ['MLP-MLP','LSTM-MLP','CNN-MLP']:
    for step,x in tqdm(enumerate(train_dataloader)):
        with torch.no_grad():
            z=model.get_latent(x).tolist()
            z_s+=z

with open(f'./tmp/latent_z/{model_type}_{year}_{month}.json','w') as f:
    json.dump(z_s,f)

# compute reconstruction error
reconstruction_error=0.0
for step,x in tqdm(enumerate(test_dataloader)):
    with torch.no_grad():
        error=model.compute_reconstruct_error(x)
        reconstruction_error+=error.item()
reconstruction_error/=(step+1)
print('>>Reconstruction Error Computed.')

with open(f'./tmp/reconstruction_error/{model_type}.txt','a') as f:
    f.write('{}_{} {} {}\n'.format(year,month,reconstruction_error,entropy_index))
f.close()
print('>>Done.')










