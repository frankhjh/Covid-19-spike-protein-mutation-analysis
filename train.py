from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from model.vae.encoder.lstm_encoder import Gaussian_LSTM_encoder1,Gaussian_LSTM_encoder2
from model.vae.encoder.cnn_encoder import multi_kernel_cnn
from model.vae.encoder.mlp_encoder import MLP_encoder
from model.vae.decoder.lstm_decoder import LSTM_decoder
from model.vae.decoder.mlp_decoder import MLP_decoder
from model.vae.VAE import *
from model.vae.model_config import model_parameters


def train_vae(model_type,model_parameters,train_dataloader,val_dataloader,epochs,lr,device,year,month,continent_name=None):
    paras=model_parameters.get(model_type)
    if model_type=='LSTM-LSTM':
        encoder=Gaussian_LSTM_encoder2(embed_dim=paras.get('embed_dim'),
                                       vocab_size=paras.get('vocab_size'),
                                       hid_size=paras.get('hid_size'),
                                       dim_z=paras.get('dim_z'),
                                       tau=paras.get('tau'))
        decoder=LSTM_decoder(dim_z=paras.get('dim_z'),
                             hid_size=paras.get('hid_size'),
                             num_aa_types=paras.get('num_aa_types'))
        model=vae_gaussian_lstm(encoder=encoder,decoder=decoder)
    
    if model_type=='LSTM-MLP':
        encoder=Gaussian_LSTM_encoder1(embed_dim=paras.get('embed_dim'),
                                       vocab_size=paras.get('vocab_size'),
                                       hid_size=paras.get('hid_size'),
                                       dim_z=paras.get('dim_z'),
                                       tau=paras.get('tau'))
        decoder=MLP_decoder(dim_z=paras.get('dim_z'),
                            num_aa_types=paras.get('num_aa_types'),
                            seq_len=paras.get('seq_len'),
                            hidden_units=paras.get('hidden_units'))
        model=vae_gaussian_mlp(encoder=encoder,decoder=decoder)
    
    if model_type=='CNN-MLP':
        encoder=multi_kernel_cnn(embed_dim=paras.get('embed_dim'),
                                 vocab_size=paras.get('vocab_size'),
                                 feat_size=paras.get('feat_size'),
                                 kernel_sizes=paras.get('kernel_sizes'),
                                 dim_z=paras.get('dim_z'),
                                 tau=paras.get('tau'))
        decoder=MLP_decoder(dim_z=paras.get('dim_z'),
                            num_aa_types=paras.get('num_aa_types'),
                            seq_len=paras.get('seq_len'),
                            hidden_units=paras.get('hidden_units'))
        model=vae_gaussian_mlp(encoder=encoder,decoder=decoder)
        
    
    if model_type=='MLP-MLP':
        encoder=MLP_encoder(dim_z=paras.get('dim_z'),
                            num_aa_types=paras.get('num_aa_types'),
                            seq_len=paras.get('seq_len'),
                            hidden_units=paras.get('hidden_units'),
                            tau=paras.get('tau'))
        decoder=MLP_decoder(dim_z=paras.get('dim_z'),
                            num_aa_types=paras.get('num_aa_types'),
                            seq_len=paras.get('seq_len'),
                            hidden_units=paras.get('hidden_units'))
        model=vae_gaussian_mlp(encoder=encoder,decoder=decoder)
    
    model=model.to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr)

    def evalute(model,val_dataloader,device):
        val_loss=0.0
        for step,x in enumerate(val_dataloader):
            x=x.to(device)
            with torch.no_grad():
                loss=model.compute_loss(x)
                val_loss+=loss.item()
        return val_loss/(step+1)
    
    min_loss,best_epoch=1e9,1
    print('>>Starting Training...')
    for epoch in range(1,epochs+1):
        total_loss=0.0
        for step,x in tqdm(enumerate(train_dataloader)):
            x=x.to(device)
            loss=model.compute_loss(x)

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            total_loss+=loss.item()
        avg_loss=total_loss/(step+1)

        val_loss=evalute(model,val_dataloader,device)
        if val_loss<min_loss:
            min_loss=val_loss
            best_epoch=epoch
            if not continent_name:
                torch.save(model.state_dict(),f'./tmp/{model_type}/bm_{year}_{month}.ckpt')
            else:
                torch.save(model.state_dict(),f'./tmp/continent_ana/{continent_name}/{model_type}/bm_{year}_{month}.ckpt')
        
        print('epoch {}, training loss:{}'.format(epoch,avg_loss)+' validation loss:{}'.format(val_loss))
    print('>>Training Done!')   
    return model



        

        
