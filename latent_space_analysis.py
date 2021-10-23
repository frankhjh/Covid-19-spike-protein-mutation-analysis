import json
import os
import random
import matplotlib.pyplot as plt
import argparse

parser=argparse.ArgumentParser(description='latent space explore')
parser.add_argument('--dim1',type=int)
parser.add_argument('--dim2',type=int)
parser.add_argument('--sample_size',type=int)
parser.add_argument('--from_model',type=str)
args=parser.parse_args()

# select 2 dim for visualizaton
def latent_space_visualization(dim1,dim2,from_model,sample_size): # dim1, dim2 are 2 dim from 128 dim
    base_path='./tmp/latent_z'

    out_dict={}

    for month in range(3,13):
        path=os.path.join(base_path,f'{from_model}_2020_{month}.json')
        with open(path,'r') as f:
            Latent_dims=json.load(f)
        res=[(latent_dims[dim1],latent_dims[dim2]) for latent_dims in Latent_dims]
        randn_idx=random.sample(range(len(res)),sample_size)
        out_dict[f'2020_{month}']=[res[idx] for idx in randn_idx]
    
    for month in range(1,7):
        path=os.path.join(base_path,f'{from_model}_2021_{month}.json')
        with open(path,'r') as f:
            Latent_dims=json.load(f)
        res=[(latent_dims[dim1],latent_dims[dim2]) for latent_dims in Latent_dims]
        randn_idx=random.sample(range(len(res)),sample_size)
        out_dict[f'2021_{month}']=[res[idx] for idx in randn_idx]
    
    # plot the data in out_dict
    # in two figs one inlcudes '2020-03' to '2020-10', the other includes '2020-11' to '2021-6'
    fig,axs=plt.subplots(nrows=1,ncols=2,facecolor='w', figsize=(17,5))
    colors=['black','brown','blue','purple','green','yellow','red','orange']
    
    ax=axs[0]
    ax.set_title('latent space variation from 2020-03 to 2020-10')
    ax.set_xlabel('dim1')
    ax.set_ylabel('dim2')

    for idx,month in enumerate(range(3,11)):
        x1=[x[0] for x in out_dict[f'2020_{month}']]
        x2=[x[1] for x in out_dict[f'2020_{month}']]
        ax.scatter(x1,x2,color=colors[idx],label=f'2020_{month}')
    ax=plt.gca()
    ax.legend()

    ax=axs[1]
    ax.set_title('latent space variation from 2020-11 to 2021-06')
    ax.set_xlabel('dim1')
    ax.set_ylabel('dim2')

    for idx,month in enumerate(range(11,13)):
        x1=[x[0] for x in out_dict[f'2020_{month}']]
        x2=[x[1] for x in out_dict[f'2020_{month}']]
        ax.scatter(x1,x2,color=colors[idx],label=f'2020_{month}')
    
    for idx,month in enumerate(range(1,7)):
        x1=[x[0] for x in out_dict[f'2021_{month}']]
        x2=[x[1] for x in out_dict[f'2021_{month}']]
        ax.scatter(x1,x2,color=colors[idx+2],label=f'2021_{month}')
    ax=plt.gca()
    ax.legend()

    plt.savefig(f'./tmp/plot_out/{from_model}_latent_plot.png')


if __name__=='__main__':
    dim1=args.dim1
    dim2=args.dim2
    sample_size=args.sample_size
    from_model=args.from_model

    latent_space_visualization(dim1,dim2,from_model,sample_size)






