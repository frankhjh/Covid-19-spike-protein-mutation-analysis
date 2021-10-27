from utils.site_entropy import site_entropy_in_group
from data_process import letter2idx
from tqdm import tqdm
import pandas as pd
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


month_name={1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',
    9:'sep',10:'oct',11:'nov',12:'dec'}

data_store=pd.HDFStore('./data/month_unique_data.h5')

if __name__=='__main__':
    sample_all=[]
    for m in tqdm(range(3,13)):
        key=month_name[m]+'2020'
        sample_m=data_store[key]
        sample_all.append(sample_m)
    
    for m in tqdm(range(1,7)):
        key=month_name[m]+'2021'
        sample_m=data_store[key]
        sample_all.append(sample_m)
    
    data_store.close()

    sample_all_df=pd.concat(sample_all).reset_index(drop=True)

    with open('./utils/mapping.json','r') as f:
        mapping_dict=json.load(f)
    
    sample_all_df=letter2idx(sample_all_df,mapping_dict)
    print('>>Sequence Encoding Done.')
    overall_site_entropy=site_entropy_in_group(sample_all_df,seq_len=1273)
    print('>>Site Entropy Calculated.')
    
    with open('./utils/overall_site_entropy.json','w') as f:
        res=[overall_site_entropy]
        json.dump(res,f)



