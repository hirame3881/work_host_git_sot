import json
from tqdm import tqdm
import numpy as np
import os,sys
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from tensorboard_logger import configure, log_value

from ntm.ntm_lstm import NTM
from ntm.datasets import CopyDataset, RepeatCopyDataset, AssociativeDataset, NGram, PrioritySort #,Nth_farthestDataset
from ntm.args import get_parser

import slackweb



parser = get_parser()
parser.add_argument('--infer_type', default=0)
parser.add_argument('--sort_flag', action='store_true')
parser.add_argument('--lstm_flag', action='store_true')
parser.add_argument('--device', default="cpu",help="cpu, cuda or int number")
parser.add_argument('-num_iters', type=int, default=100000,
                    help='number of iterations for training')
parser.add_argument('-summarize_freq', type=int, default=200,
                    help='iter/freq calculate acc , save score')
parser.add_argument('-runid', type=int, default=1,
                    help='same setting/model run count')
args=parser.parse_args()
print("infer:",args.infer_type)
print("sort:",args.sort_flag)

#------
sys.path.append("/work/my_modified_ntm/bgavDNC_src")
from task_implementations.bAbI.bAbI import *
from utils import project_path, init_wrapper

sys.path.append("/work/handmade_utils/sotsuron_scores")
with __import__('importnb').Notebook(): 
    from score_store import ScoreStoring,get_dirname

config_file="./ntm/tasks/config_all.json"
with open(config_file, "r") as fp:
    config = json.load(fp)
task_params = json.load(open(args.task_json))
rel_config=task_params["rrnnconfig"]
data_config=task_params["data"]

#sys.path.append("/work/handmade_utils/sotsuron_scores")
file_description={"rowAttr":"iter","rowVal":"errorrate","batch_size":args.batch_size}
dir_modelname=get_dirname(args.infer_type,args.sort_flag)
scorestoring =ScoreStoring(data_config,data_config["savefileid_list"],dir_modelname,args.runid,file_description,babi=False)


#------
batch_size=args.batch_size
device = torch.device(args.device)

#configure("runs/")

# ----------------------------------------------------------------------------
# -- initialize datasets, model, criterion and optimizer
# ----------------------------------------------------------------------------

#args.task_json = 'ntm/tasks/copy.json'
'''
args.task_json = 'ntm/tasks/repeatcopy.json'
args.task_json = 'ntm/tasks/associative.json'
args.task_json = 'ntm/tasks/ngram.json'
args.task_json = 'ntm/tasks/prioritysort.json'
'''
#num_dims =task_params['seq_width']
#num_vectors =task_params['input_seq_len']

def mycross_OH_3Dto3D(input, target, reduction='mean'):
    if reduction!="mean": raise ValueError('reduction is not implemented yet')
    if (input.dim()!=3) or (target.dim()!=3): raise ValueError('invalid dim')
    s_input=nn.functional.softmax(input, dim=2)
    ls_input=torch.log(s_input+1e-7)
    nll=-1*torch.einsum( 'blf,blf->bl',ls_input,target) 
    #nll=torch.mean(nll,dim=1)
    return nll
def my_masked_mean(loss,mask):
    masked = loss * mask[:, :, 0]
    mask_batchs =torch.sum(mask[:,:,0],dim=1)
    return torch.mean(torch.sum(masked,dim=1)/ mask_batchs)


task = bAbITask(os.path.join("tasks_1-20_v1-2", "en-10k"))
print("Loaded task")

'''
dataset = CopyDataset(task_params)
dataset = RepeatCopyDataset(task_params)
dataset = AssociativeDataset(task_params)
dataset = NGram(task_params)

'''

"""
For the Copy task, input_size: seq_width + 2, output_size: seq_width
For the RepeatCopy task, input_size: seq_width + 2, output_size: seq_width + 1
For the Associative task, input_size: seq_width + 2, output_size: seq_width
For the NGram task, input_size: 1, output_size: 1
For the Priority Sort task, input_size: seq_width + 1, output_size: seq_width
Nth 
"""
ntm = NTM(input_size=task.vector_size,
          output_size=task.vector_size,
          controller_size=task_params['controller_size'],
          memory_units=task_params['memory_units'],
          memory_unit_size=task_params['memory_unit_size'],
          num_heads=task_params['num_heads'],
          rel_config=rel_config,
          device=device,
          infer_flag=args.infer_type,
          sort_flag=args.sort_flag,
        batch_size=args.batch_size,
        softmax=False,
        lstm_flag=args.lstm_flag
        ).to(device)
print(ntm)
total_params = sum(p.numel() for p in ntm.parameters() if p.requires_grad)
print("Model built, total trainable params: " + str(total_params))
###criterion = mycross_OH_3Dto3D
criterion = nn.CrossEntropyLoss()
# As the learning rate is task specific, the argument can be moved to json file
'''
optimizer = optim.RMSprop(ntm.parameters(),
                          lr=args.lr,
                          alpha=args.alpha,
                          momentum=args.momentum)
'''
optimizer = optim.Adam(ntm.parameters(), lr=args.lr,
                       betas=(args.beta1, args.beta2))
##scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=8e-5) ##?

#args.saved_model = 'saved_model_copy.pt'
'''
args.saved_model = 'saved_model_repeatcopy.pt'
args.saved_model = 'saved_model_associative.pt'
args.saved_model = 'saved_model_ngram.pt'
args.saved_model = 'saved_model_prioritysort.pt'
'''

cur_dir = os.getcwd()
PATH = os.path.join(cur_dir, args.saved_model)


# ----------------------------------------------------------------------------
# -- basic training loop
# ----------------------------------------------------------------------------
losses = []
#errors = []
a91_flag=False
for iter in tqdm(range(args.num_iters)):
    optimizer.zero_grad()
    ntm.reset(batch_size=args.batch_size)

    data_batch,seqlen, m =task.generate_data(batch_size=batch_size,train=True)
    input =torch.tensor(data_batch[0],dtype=torch.float).to(device)
    
    ###target=torch.tensor(data_batch[1],dtype=torch.float).to(device)
    target_np=np.argmax(data_batch[1],axis=2)
    target=torch.tensor(target_np).to(device)
    ###mask=torch.tensor(m).to(device,dtype=torch.float)
    mask=torch.tensor(m==1).to(device,dtype=torch.bool)
    masked_target1D=torch.masked_select(target,mask[:,:,0])

    if iter==1: print("input_size:",input.size(),"target_size:",target.size(),"mask_size:",mask.size())
    ##out = torch.zeros(target.size()).to(device)

    # -------------------------------------------------------------------------
    # loop for other tasks
    # -------------------------------------------------------------------------
    outs=torch.zeros(input.size()).to(device) ###target.size()).to(device)
    for i in range(input.size()[1]):##
        # to maintain consistency in dimensions as torch.cat was throwing error
        ##in_data = torch.unsqueeze(input[i], 0)
        in_data = input[:,i,:]##(batch_size,in_feature)になるが、controllerのLSTMCell入力がちょうどこれなので問題なし
        #print(in_data.dtype,outs[:,i,:].dtype)
        outs[:,i,:]=ntm(in_data)
    masked_outs=torch.masked_select(outs,mask).view(-1,task.vector_size)


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # loop for NGram task
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------

    ###loss = criterion(outs, target)
    loss=criterion( masked_outs,masked_target1D)
    ##loss=torch.mean(loss * mask[:, :, 0])
    ###loss =my_masked_mean(loss,mask)
    losses.append(loss.item())
    loss.backward()
    # clips gradient in the range [-10,10]. Again there is a slight but
    # insignificant deviation from the paper where they are clipped to (-10,10)
    ##nn.utils.clip_grad_value_(ntm.parameters(), 10)
    nn.utils.clip_grad_norm_(ntm.parameters(), 50)
    optimizer.step()

    ##y_pred = torch.argmax(outs.clone().detach(), dim=2)
    ##acc = accuracy_score_babi(y_pred.cpu(), target.cpu(),mask)
    ##errors.append(acc.item())

    # ---logging---
    if iter % (args.summarize_freq/10) == 0 and iter!=0:
        print('Iteration: %d\ttrain_Loss: %.2f\t /: %.2f' %
              (iter, np.mean(losses), 0))
        losses = []
        #log_value('train_loss', np.mean(losses), iter)
        #log_value('bit_error_per_sequence', np.mean(errors), iter)
        
        # ========== TEST ALL TASKS  for accuracy================
        
    if iter % (args.summarize_freq) == 0 and iter!=0:
        test_losses=[]
        test_errors = []
        task.test(ntm,batch_size=1,device=device)

        '''scorestoring.store(data_config["savefileid_list"][0],iter,np.mean(test_errors))
        if np.mean(test_errors)>0.70 and (not a91_flag):
            slack = slackweb.Slack(url="https://hooks.slack.com/services/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")
            slack.notify(text="over 91% !")
            a91_flag=True'''
    if np.isnan(loss.item()):
        print("nan loss")
        break

slack = slackweb.Slack(url=config["slackurl"])
slack.notify(text="done:" + str(os.path.basename(__file__)) +" "+ dir_modelname + " runid:"+str(args.runid))

# ---saving the model---
torch.save(ntm.state_dict(), PATH)
# torch.save(ntm, PATH)

#test 入れるときに使う
def get_batch(X, y, batch_num, batch_size=32, batch_first=True):
    if not batch_first:
        raise NotImplementedError
    start = batch_num * batch_size
    end = (batch_num + 1) * batch_size
    return X[start:end], y[start:end]