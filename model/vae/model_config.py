
model_parameters={
    'LSTM-LSTM':{'embed_dim':128,
                 'vocab_size':22,
                 'hid_size':128,
                 'dim_z':128,
                 'tau':0.5,
                 'num_aa_types':22},
    'LSTM-MLP':{'embed_dim':128,
                'vocab_size':22,
                'hid_size':128,
                'dim_z':128,
                'tau':0.5,
                'num_aa_types':22,
                'seq_len':1264,
                'hidden_units':[512,1024]}
    'MLP-MLP':{'dim_z':128,
               'num_aa_types':22,
               'seq_len':1264,
               'hidden_units':[512,1024],
               'tau':0.5}
}