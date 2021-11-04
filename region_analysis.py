#!/usr/bin/env python
from region_info import countries
from btw_group_diff_measure import cal_group_kl_divergence,cal_group_diff
import pandas as pd
from data_process import letter2idx
from tqdm import tqdm
import argparse
import json
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


parser=argparse.ArgumentParser(description='region_analysis')
parser.add_argument('--year',type=int)
parser.add_argument('--month',type=int)
parser.add_argument('--continent',type=str)
parser.add_argument('--method',type=str)
args=parser.parse_args()



year=args.year
month=args.month
continent=args.continent
method=args.method
print(year)
print(month)
print(continent)

def continent_split(df,continent,countries_dict):
    sub_df=df[df.country.isin(countries_dict[continent])]
    return sub_df.reset_index(drop=True)


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

# foucs on one continent
sub_df1=continent_split(df1,continent,countries)
sub_df2=continent_split(df2,continent,countries)

with open('./utils/mapping.json','r') as f:
    mapping_dict=json.load(f)

current_month=letter2idx(sub_df1,mapping_dict)
next_month=letter2idx(sub_df2,mapping_dict)
print('>>Sequence Encoding Done.')


aa_idxs=list(mapping_dict.values())

if method=='kl_divergence':
    avg_kld=0.0
    for i in range(5):
        kld=cal_group_kl_divergence(current_month,next_month,sample_size=5000,seq_len=1273,aa_idxs=aa_idxs)
        avg_kld+=kld/5
    
    file_path=f'./tmp/region_based_ana/{continent}/kl_divergence.txt' if continent in ['Africa','Europe','Asia'] else \
        f'./tmp/region_based_ana/{"_".join(continent.split())}/kl_divergence.txt'

    with open(file_path,'a') as f:
        f.write('{}_{} {}\n'.format(year,month,avg_kld))
    f.close()

elif method=='diff_ratio':
    diff_ratio=cal_group_diff(current_month,next_month,sample_size=1000)

    file_path=f'./tmp/region_based_ana/{continent}/diff_ratio.txt' if continent in ['Africa','Europe','Asia'] else \
        f'./tmp/region_based_ana/{"_".join(continent.split())}/diff_ratio.txt'
    
    with open(file_path,'a') as f:
        f.write('{}_{} {}\n'.format(year,month,diff_ratio))
    f.close()

print('>>Done.')
