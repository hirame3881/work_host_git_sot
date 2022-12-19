import os, sys
import json
PATH_CUR = os.path.dirname(os.path.realpath("__file__"))
sys.path.append(PATH_CUR)
with __import__('importnb').Notebook(): 
    from score_store import ScoreStoringLoader
import pandas as pd
import matplotlib.pyplot as plt


def get_multiSSasDF(readSS_config,fileid_str_list):
    dir_data_dic ={}
    for dirname in readSS_config["read_directories"]:
        ssl=ScoreStoringLoader(dirname,readSS_config["filename_common"],fileid_str_list)
        dir_data_dic[dirname] =ssl.get_data_dic()
    df=pd.DataFrame(dir_data_dic).reindex(index=fileid_str_list)
    return df


with open("./config_sotsuron.json", "r") as fp:
    config = json.load(fp)
readSS_config=config["read_ScoreStoring"]
vis_config =config["visualization"]

fileid_str_list =[str(i) for i in range(1,21)]
fileid_str_list.append("mean")

accSequence_df =get_multiSSasDF(readSS_config,fileid_str_list)



plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
for task,data in accSequence_df.iterrows():
    png_path=os.path.join(vis_config["savedir"],vis_config["savefile_common"]+task+".png")

    fig1 = plt.figure(figsize=(512,512))


    ax1 = fig1.add_subplot(111)
    ax1.set_xlim(left=0)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("Epoch",fontsize=14)
    ax1.set_ylabel("Accuracy",fontsize=14)


    for ind,label in enumerate(readSS_config["read_directories"]):
        x=data[label]["epoch"]
        y=data[label]["accuracy"]
        ax1.plot(x,y,marker=".",label=vis_config["model_labels"][ind],markersize=4) #ls linestyle

    handles, labels = ax1.get_legend_handles_labels()
    fig1.legend(handles, labels,title="models",fontsize=13,loc='upper left',)
    #plt.legend(h1+h2, l1+l2)
    fig1.suptitle('task = '+task)
    
    fig1.savefig(png_path,bbox_inches="tight")
