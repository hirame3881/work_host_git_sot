import json
from tqdm import tqdm
import numpy as np
import os,sys

import torch
from torch import nn, optim
from tensorboard_logger import configure, log_value

from ntm import NTM
from ntm.datasets import CopyDataset, RepeatCopyDataset, AssociativeDataset, NGram, PrioritySort
from ntm.args import get_parser

import slackweb



parser = get_parser()
parser.add_argument('--task_type')
parser.add_argument('--seq_len', type=int, default=-1,
                    help='for priority')
parser.add_argument('--infer_type', type=int,default=0)
parser.add_argument('--sort_flag', action='store_true')
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
sys.path.append("/work/handmade_utils/sotsuron_scores")
with __import__('importnb').Notebook(): 
    from score_store import ScoreStoring,get_dirname

config_file="./ntm/tasks/config_all.json"
with open(config_file, "r") as fp:
    config = json.load(fp)
task_params = json.load(open(args.task_json))
if args.task_type=="priority" and args.seq_len!=-1:
    task_params["input_seq_len"]=args.seq_len
    for idx,_ in enumerate( task_params["data"]["savefileid_list"]):
        task_params["data"]["savefileid_list"][idx] +="ms"+str(task_params["memory_units"])
        task_params["data"]["savefileid_list"][idx] +="l"+str(task_params["input_seq_len"])

print("input_seq_len:",task_params["input_seq_len"])

rel_config=task_params["rrnnconfig"]
#data_config=config["data"]
data_config=task_params["data"]

sys.path.append("/work/handmade_utils/sotsuron_scores")
file_description={"rowAttr":"iter","rowVal":"bitError20","batch_size":args.batch_size}
if args.task_type=="priority":
    file_description["input_seq_len"]=task_params["input_seq_len"]
#dir_modelname=get_dirname(args.infer_type,args.sort_flag)
dir_modelname="SAM"
scorestoring =ScoreStoring(data_config,data_config["savefileid_list"],dir_modelname,args.runid,file_description)
#------
batch_size=args.batch_size
device = torch.device(args.device)


if args.task_type=="associative":
    dataset = AssociativeDataset(task_params)
elif args.task_type=="priority":
    dataset = PrioritySort(task_params)


if args.task_type=="associative":
    ntm_input_size=task_params['seq_width'] + 2
elif args.task_type=="priority":
    ntm_input_size=task_params['seq_width'] + 1

sys.path.append("/work/modified_SAM-master")
from samimport~

ntm=SAM(input_size=ntm_input_size,
          output_size=task_params['seq_width'],
          controller_size=task_params['controller_size'],
          memory_units=task_params['memory_units'],
          memory_unit_size=task_params['memory_unit_size'],
          num_heads=task_params['num_heads'],
          rel_config=rel_config,
          device=device,
          infer_flag=args.infer_type,
          sort_flag=args.sort_flag,
        batch_size=args.batch_size
        ).to(device)

ntm = NTM(input_size=ntm_input_size,
          output_size=task_params['seq_width'],
          controller_size=task_params['controller_size'],
          memory_units=task_params['memory_units'],
          memory_unit_size=task_params['memory_unit_size'],
          num_heads=task_params['num_heads'],
          rel_config=rel_config,
          device=device,
          infer_flag=args.infer_type,
          sort_flag=args.sort_flag,
        batch_size=args.batch_size
        ).to(device)
print(ntm)
total_params = sum(p.numel() for p in ntm.parameters() if p.requires_grad)
print("Model built, total trainable params: " + str(total_params))
criterion = nn.BCELoss()
# As the learning rate is task specific, the argument can be moved to json file

optimizer = optim.Adam(ntm.parameters(), lr=args.lr,
                       betas=(args.beta1, args.beta2))

'''
args.saved_model = 'saved_model_associative.pt'
args.saved_model = 'saved_model_prioritysort.pt'
'''

cur_dir = os.getcwd()
PATH = os.path.join(cur_dir, args.saved_model)

# ----------------------------------------------------------------------------
# -- basic training loop
# ----------------------------------------------------------------------------
losses = []
errors = []
for iter in tqdm(range(args.num_iters)):
    optimizer.zero_grad()
    ntm.reset(batch_size=args.batch_size)

    if args.task_type=="associative":
        #data = dataset[iter]
        #input, target = data['input'].to(device).repeat(batch_size,1,1) , data['target'].to(device).repeat(batch_size,1,1)  ##
        dataset.set_numitem()
        input,target=[],[]
        for b in range(batch_size):
            data = dataset[iter]
            input.append(data['input'] )
            target.append(data['target'] )
        input=torch.stack(input).to(device)
        target=torch.stack(target).to(device)
    elif args.task_type=="priority":
        input,target=[],[]
        for b in range(batch_size):
            data = dataset[iter]
            input.append(data['input'] )
            target.append(data['target'] )
        input=torch.stack(input).to(device)
        target=torch.stack(target).to(device)
    if iter==1: print("input_size:",input.size(),"target_size:",target.size())
    out = torch.zeros(target.size()).to(device)

    # -------------------------------------------------------------------------
    # loop for other tasks
    # -------------------------------------------------------------------------
    for i in range(input.size()[1]):##
        # to maintain consistency in dimensions as torch.cat was throwing error
        ##in_data = torch.unsqueeze(input[i], 0)
        in_data = input[:,i,:]##(batch_size,in_feature)になるが、controllerのLSTMCell入力がちょうどこれなので問題なし
        ntm(in_data)



    # passing zero vector as input while generating target sequence
    ##in_data = torch.unsqueeze(torch.zeros(input.size()[1]), 0).to(device)
    in_data = torch.zeros(input.size()[0],input.size()[2]).to(device)
    for i in range(target.size()[1]):
        out[:,i,:] = ntm(in_data)
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # loop for NGram task
    # -------------------------------------------------------------------------
    '''
    for i in range(task_params['seq_len'] - 1):
        in_data = input[i].view(1, -1)
        ntm(in_data)
        target_data = torch.zeros([1]).view(1, -1)
        out[i] = ntm(target_data)
    '''
    # -------------------------------------------------------------------------

    loss = criterion(out, target)
    losses.append(loss.item())
    loss.backward()
    # clips gradient in the range [-10,10]. Again there is a slight but
    # insignificant deviation from the paper where they are clipped to (-10,10)
    ##nn.utils.clip_grad_value_(ntm.parameters(), 10)
    nn.utils.clip_grad_norm_(ntm.parameters(), 50)
    optimizer.step()

    binary_output = out.clone()
    binary_output = binary_output.detach().cpu().apply_(lambda x: 0 if x < 0.5 else 1) ##

    # sequence prediction error is calculted in bits per sequence
    error = torch.sum(torch.abs(binary_output - target.cpu()))/batch_size
    errors.append(error.item())

    # ---logging---
    if iter % args.summarize_freq == 0:
        errorP20 =(np.mean(errors)/task_params["input_seq_len"])*20
        print('Iteration: %d\tLoss: %.2f\tError in bits per 20sequence: %.2f' %
              (iter, np.mean(losses), errorP20))
        #log_value('train_loss', np.mean(losses), iter)
        #log_value('bit_error_per_sequence', np.mean(errors), iter)
        scorestoring.store(data_config["savefileid_list"][0],iter,errorP20)
        if  errorP20<1.0:
            print("error bits per 20 < 1.0")
            break
        losses = []
        errors = []

#slack = slackweb.Slack(url=config["slackurl"])
#slack.notify(text="done:" + str(os.path.basename(__file__)) +" "+ dir_modelname + " runid:"+str(args.runid))

# ---saving the model---
torch.save(ntm.state_dict(), PATH)
# torch.save(ntm, PATH)
