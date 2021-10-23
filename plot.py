import matplotlib.pyplot as plt

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




