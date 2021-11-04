import matplotlib.pyplot as plt
from collections import defaultdict

def reconstruction_error_plot(model_type,color,continent_name=None):
    if not continent_name:
        months=[]
        errors=[]
        site_entropys=[]
        with open(f'./tmp/reconstruction_error/{model_type}.txt','r') as f:
            for line in f:
                info=line.strip().split(' ')
                months.append(info[0])
                errors.append(info[1])
                site_entropys.append(info[2])
        f.close()
        errors=[float(error) for error in errors]
        site_entropys=[float(site_entropy) for site_entropy in site_entropys]

        fig,axs=plt.subplots(nrows=1,ncols=2,facecolor='w', figsize=(17,5))

        ax=axs[0]
        ax.set_title(f'reconstruction error variation under {model_type}')
        ax.plot(months,errors,color=color[0])
        ax.set_xlabel('month')
        ax.set_ylabel('reconstruction error')
        ax.set_xticklabels(labels=months,rotation=270)

        ax=axs[1]
        ax.set_title(f'site entropy variation')
        ax.plot(months,site_entropys,color=color[1])
        ax.set_xlabel('month')
        ax.set_ylabel('site entropy')
        ax.set_xticklabels(labels=months,rotation=270)
        
        plt.savefig(f'./tmp/plot_out/{model_type} reconstruction_error & site_entropy.png')

def reconstruction_error_single_plot(model_type,color):
        months=[]
        errors=[]
        with open(f'./tmp/reconstruction_error/{model_type}.txt','r') as f:
            for line in f:
                info=line.strip().split(' ')
                months.append(info[0])
                errors.append(info[1])
        f.close()
        errors=[float(error) for error in errors]

        fig,axs=plt.subplots(nrows=1,ncols=1,facecolor='w', figsize=(12,7))

        ax=axs
        ax.set_title(f'reconstruction error variation under {model_type}')
        ax.plot(months,errors,color=color)
        ax.set_xlabel('month')
        ax.set_ylabel('reconstruction error')
        ax.set_xticklabels(labels=months,rotation=270)
        
        plt.savefig(f'./tmp/plot_out/{model_type} reconstruction_error.png')

def diff_measure_plot(method,color):
    months=[]
    measures=[]
    with open(f'./tmp/diff_measure/{method}.txt','r') as f:
        for line in f:
            info=line.strip().split(' ')
            months.append(info[0])
            measures.append(info[1])
    f.close()
    measures=[float(i) for i in measures]

    fig,axs=plt.subplots(nrows=1,ncols=1,facecolor='w', figsize=(12,7))

    ax=axs
    ax.set_title(f'between-month mutation measure by {method}')
    ax.plot(months,measures,color=color)
    ax.set_xlabel('month')
    ax.set_ylabel(f'{method}')
    ax.set_xticklabels(labels=months,rotation=270)
    
    plt.savefig(f'./tmp/plot_out/{method}.png')


# region analysis
def diff_measure_plot_comb(methods,continent,color):
    months=[]
    measures1=[]
    measures2=[]
    base_path='./tmp/region_based_ana/'
    path1=os.path.join(base_path,f'./{continent}/{methods[0]}.txt')
    path2=os.path.join(base_path,f'./{continent}/{methods[1]}.txt')
    with open(path1,'r') as f1:
        for line in f1:
            info=line.strip().split(' ')
            months.append(info[0])
            measures1.append(float(info[1]))
    f1.close()
    
    with open(path2,'r') as f2:
        for line in f2:
            info=line.strip().split(' ')
            measures2.append(float(info[1]))
    f2.close()
      

    fig,axs=plt.subplots(nrows=1,ncols=2,facecolor='w', figsize=(17,5))

    ax=axs[0]
    ax.set_title(f'monthly mutation for {continent} measured by {methods[0]}')
    ax.plot(months,measures1,color=color[0])
    ax.set_xlabel('month')
    ax.set_ylabel(f'{methods[0]}')
    ax.set_xticklabels(labels=months,rotation=270)
    
    ax=axs[1]
    ax.set_title(f'monthly mutation for {continent} measured by {methods[1]}')
    ax.plot(months,measures2,color=color[1])
    ax.set_xlabel('month')
    ax.set_ylabel(f'{methods[1]}')
    ax.set_xticklabels(labels=months,rotation=270)
    
    plt.savefig(f'./tmp/plot_out/{continent}.png')


def regions_compare(method,colors):
    months=[]
    
    continents_measure=defaultdict(list)
    
    base_path='./Covid-19-spike-protein-mutation-analysis/tmp/region_based_ana/'
    path1=os.path.join(base_path,f'./Asia/{method}.txt')
    path2=os.path.join(base_path,f'./Europe/{method}.txt')
    path3=os.path.join(base_path,f'./North_America/{method}.txt')
    path4=os.path.join(base_path,f'./South_America/{method}.txt')
    path5=os.path.join(base_path,f'./Africa/{method}.txt')
    
    with open(path1,'r') as f1:
        for line in f1:
            info=line.strip().split(' ')
            months.append(info[0])
            continents_measure['Asia'].append(float(info[1]))
    f1.close()
    
    with open(path2,'r') as f2:
        for line in f2:
            info=line.strip().split(' ')
            continents_measure['Europe'].append(float(info[1]))
    f2.close()
    
    with open(path3,'r') as f3:
        for line in f3:
            info=line.strip().split(' ')
            continents_measure['North_America'].append(float(info[1]))
    f3.close()
    
    with open(path4,'r') as f4:
        for line in f4:
            info=line.strip().split(' ')
            continents_measure['South_America'].append(float(info[1]))
    f4.close()
    
    with open(path5,'r') as f5:
        for line in f5:
            info=line.strip().split(' ')
            continents_measure['Africa'].append(float(info[1]))
    f5.close()
      

    fig,axs=plt.subplots(nrows=1,ncols=1,facecolor='w', figsize=(12,7))

    ax=axs
    ax.set_title(f'Monthly mutation for different continents measured by {method}')
    ax.plot(months,continents_measure['Asia'],color=colors[0],label='Asia')
    ax.plot(months,continents_measure['Europe'],color=colors[1],label='Europe')
    ax.plot(months,continents_measure['North_America'],color=colors[2],label='North America')
    ax.plot(months,continents_measure['South_America'],color=colors[3],label='South America')
    ax.plot(months,continents_measure['Africa'],color=colors[4],label='Africa')
    
    ax.set_xlabel('month')
    ax.set_ylabel(f'{method}')
    ax.set_xticklabels(labels=months,rotation=270)
    
    ax=plt.gca()
    ax.legend()
    
    plt.savefig(f'.tmp/plot_out/{method}_continent_compare.png')


def baseline_compare(colors):
    months=[]
    errors_baseline=[]
    errors_standard=[]
    
    base_path='./tmp/reconstruction_error/'
    path1=os.path.join(base_path,f'./MLP-MLP_baseline.txt')
    path2=os.path.join(base_path,f'./MLP-MLP.txt')
    
    with open(path1,'r') as f1:
        for line in f1:
            info=line.strip().split(' ')
            months.append(info[0])
            errors_baseline.append(float(info[1]))
            
    f1.close()
    
    with open(path2,'r') as f2:
        for line in f2:
            info=line.strip().split(' ')
            errors_standard.append(float(info[1]))
    f2.close()
    
    
    fig,axs=plt.subplots(nrows=1,ncols=1,facecolor='w', figsize=(12,7))

    ax=axs
    ax.set_title(f'Comparsion between baseline and standard MLP-MLP')
    ax.plot(months,errors_baseline,color=colors[0],label='baseline')
    ax.plot(months,errors_standard,color=colors[1],label='standard')
    
    ax.set_xlabel('month')
    ax.set_ylabel('reconstruction error')
    ax.set_xticklabels(labels=months,rotation=270)
    
    ax=plt.gca()
    ax.legend()
    
    plt.savefig(f'./tmp/plot_out/baseline_compare.png')

def different_nn_compare(colors):
    months=[]
    errors_mlp_mlp=[]
    errors_lstm_mlp=[]
    errors_cnn_mlp=[]
    errors_lstm_lstm=[]
    
    base_path='./tmp/reconstruction_error/'
    path1=os.path.join(base_path,f'./MLP-MLP.txt')
    path2=os.path.join(base_path,f'./LSTM-MLP.txt')
    path3=os.path.join(base_path,f'./CNN-MLP.txt')
    path4=os.path.join(base_path,f'./LSTM-LSTM.txt')
    
    with open(path1,'r') as f1:
        for line in f1:
            info=line.strip().split(' ')
            months.append(info[0])
            errors_mlp_mlp.append(float(info[1]))
            
    f1.close()
    
    with open(path2,'r') as f2:
        for line in f2:
            info=line.strip().split(' ')
            errors_lstm_mlp.append(float(info[1]))
    f2.close()
    
    with open(path3,'r') as f3:
        for line in f3:
            info=line.strip().split(' ')
            errors_cnn_mlp.append(float(info[1]))
    f3.close()
    
    with open(path4,'r') as f4:
        for line in f4:
            info=line.strip().split(' ')
            errors_lstm_lstm.append(float(info[1])*50)
    f4.close()
    
    
    fig,axs=plt.subplots(nrows=1,ncols=1,facecolor='w', figsize=(12,7))

    ax=axs
    ax.set_title(f'Comparsion of Different NN Architectures')
    ax.plot(months,errors_mlp_mlp,color=colors[0],label='MLP-MLP')
    ax.plot(months,errors_lstm_mlp,color=colors[1],label='LSTM-MLP')
    ax.plot(months,errors_cnn_mlp,color=colors[2],label='CNN-MLP')
    ax.plot(months,errors_lstm_lstm,color=colors[3],label='LSTM-LSTM')
    
    ax.set_xlabel('month')
    ax.set_ylabel('reconstruction error')
    ax.set_xticklabels(labels=months,rotation=270)
    
    ax=plt.gca()
    ax.legend()
    
    plt.savefig(f'.tmp/plot_out/NN_compare.png')

def final_comparsion(method,colors):
    months=[]
    errors_mlp_mlp=[]
    errors_lstm_mlp=[]
    errors_cnn_mlp=[]
    errors_lstm_lstm=[]
    errors_measure=[]
    
    base_path='./tmp/reconstruction_error/'
    path1=os.path.join(base_path,f'./MLP-MLP.txt')
    path2=os.path.join(base_path,f'./LSTM-MLP.txt')
    path3=os.path.join(base_path,f'./CNN-MLP.txt')
    path4=os.path.join(base_path,f'./LSTM-LSTM.txt')
    
    path5=os.path.join('./tmp/',f'./diff_measure/{method}.txt')
    
    with open(path1,'r') as f1:
        for line in f1:
            info=line.strip().split(' ')
            months.append(info[0])
            if method=='kl_divergence':
                errors_mlp_mlp.append(float(info[1])/100000)
            elif method=='diff_ratio':
                errors_mlp_mlp.append(float(info[1])/10000)
            
    f1.close()
    
    with open(path2,'r') as f2:
        for line in f2:
            info=line.strip().split(' ')
            if method=='kl_divergence':
                errors_lstm_mlp.append(float(info[1])/100000)
            elif method=='diff_ratio':
                errors_lstm_mlp.append(float(info[1])/10000)
    
    f2.close()
    
    with open(path3,'r') as f3:
        for line in f3:
            info=line.strip().split(' ')
            if method=='kl_divergence':
                errors_cnn_mlp.append(float(info[1])/100000)
            elif method=='diff_ratio':
                errors_cnn_mlp.append(float(info[1])/10000)
    
    f3.close()
    
    with open(path4,'r') as f4:
        for line in f4:
            info=line.strip().split(' ')
            if method=='kl_divergence':
                errors_lstm_lstm.append(float(info[1])*50/100000)
            elif method=='diff_ratio':
                errors_lstm_lstm.append(float(info[1])*50/10000)
    f4.close()
    
    with open(path5,'r') as f5:
        for line in f5:
            info=line.strip().split(' ')
            errors_measure.append(float(info[1]))
    
    f5.close()
    

    
    fig,axs=plt.subplots(nrows=1,ncols=1,facecolor='w', figsize=(12,7))

    ax=axs
    ax.set_title(f'Comparsion of Different NN Architectures')
    ax.plot(months,errors_mlp_mlp,color=colors[0],label='MLP-MLP')
    ax.plot(months,errors_lstm_mlp,color=colors[1],label='LSTM-MLP')
    ax.plot(months,errors_cnn_mlp,color=colors[2],label='CNN-MLP')
    ax.plot(months,errors_lstm_lstm,color=colors[3],label='LSTM-LSTM')
    ax.plot(months,errors_measure,color=colors[4],label=f'{method}')
    
    ax.set_xlabel('month')
    ax.set_ylabel(f'scaled reconstruction error & {method}')
    ax.set_xticklabels(labels=months,rotation=270)
    
    ax=plt.gca()
    ax.legend()
    
    plt.savefig(f'./final_compare.png')
    

    