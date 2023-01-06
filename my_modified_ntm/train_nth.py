import json
from tqdm import tqdm
import numpy as np
import os,sys
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from tensorboard_logger import configure, log_value

from ntm import NTM
from ntm.datasets import CopyDataset, RepeatCopyDataset, AssociativeDataset, NGram, PrioritySort #,Nth_farthestDataset
from ntm.args import get_parser

import slackweb



parser = get_parser()
parser.add_argument('--infer_type',type=int, default=0)
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
rel_config=task_params["rrnnconfig"]
data_config=task_params["data"]

#sys.path.append("/work/handmade_utils/sotsuron_scores")
file_description={"rowAttr":"iter","rowVal":"accuracy","batch_size":args.batch_size}
dir_modelname=get_dirname(args.infer_type,args.sort_flag)
scorestoring =ScoreStoring(data_config,data_config["savefileid_list"],dir_modelname,args.runid,file_description)
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


num_dims =task_params['seq_width']
num_vectors =task_params['input_seq_len']

#dataset = Nth_farthestDataset(task_params)
# For each example
input_size = num_dims + num_vectors * 3

def one_hot_encode(array, num_dims=8):
    one_hot = np.zeros((len(array), num_dims))
    for i in range(len(array)):
        one_hot[i, array[i]] = 1
    return one_hot
def get_example(num_vectors, num_dims):
    input_size = num_dims + num_vectors * 3
    n = np.random.choice(num_vectors, 1)  # nth farthest from target vector
    labels = np.random.choice(num_vectors, num_vectors, replace=False)
    m_index = np.random.choice(num_vectors, 1)  # m comes after the m_index-th vector
    m = labels[m_index]

    # Vectors sampled from U(-1,1)
    vectors = np.random.rand(num_vectors, num_dims) * 2 - 1
    target_vector = vectors[m_index]
    dist_from_target = np.linalg.norm(vectors - target_vector, axis=1)
    X_single = np.zeros((num_vectors, input_size))
    X_single[:, :num_dims] = vectors
    labels_onehot = one_hot_encode(labels, num_dims=num_vectors)
    X_single[:, num_dims:num_dims + num_vectors] = labels_onehot
    nm_onehot = np.reshape(one_hot_encode([n, m], num_dims=num_vectors), -1)
    X_single[:, num_dims + num_vectors:] = np.tile(nm_onehot, (num_vectors, 1))
    y_single = labels[np.argsort(dist_from_target)[-(n + 1)]]

    return X_single, y_single
def get_examples(num_examples, num_vectors, num_dims, device):
    X = np.zeros((num_examples, num_vectors, input_size))
    y = np.zeros(num_examples)
    for i in range(num_examples):
        X_single, y_single = get_example(num_vectors, num_dims)
        X[i, :] = X_single
        y[i] = y_single

    X = torch.Tensor(X).to(device)
    y = torch.LongTensor(y).to(device)

    return X, y


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
ntm = NTM(input_size=task_params['seq_width'] + task_params['input_seq_len']*3,
          output_size=task_params['input_seq_len'],
          controller_size=task_params['controller_size'],
          memory_units=task_params['memory_units'],
          memory_unit_size=task_params['memory_unit_size'],
          num_heads=task_params['num_heads'],
          rel_config=rel_config,
          device=device,
          infer_flag=args.infer_type,
          sort_flag=args.sort_flag,
        batch_size=args.batch_size,
        softmax=False
        ).to(device)
print(ntm)
total_params = sum(p.numel() for p in ntm.parameters() if p.requires_grad)
print("Model built, total trainable params: " + str(total_params))
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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=8e-5) ##?

#args.saved_model = 'saved_model_copy.pt'
'''
args.saved_model = 'saved_model_repeatcopy.pt'
args.saved_model = 'saved_model_associative.pt'
args.saved_model = 'saved_model_ngram.pt'
args.saved_model = 'saved_model_prioritysort.pt'
'''

cur_dir = os.getcwd()
PATH = os.path.join(cur_dir, args.saved_model)

def accuracy_score(y_pred, y_true):
    return np.array(y_pred == y_true).sum() * 1.0 / len(y_true)
# ----------------------------------------------------------------------------
# -- basic training loop
# ----------------------------------------------------------------------------
losses = []
errors = []
a91_flag=False
for iter in tqdm(range(args.num_iters)):
    optimizer.zero_grad()
    ntm.reset(batch_size=args.batch_size)

    data, targets = get_examples(batch_size, num_vectors, num_dims, device)
    input,target=data,targets

    if iter==1: print("input_size:",input.size(),"target_size:",target.size())
    ##out = torch.zeros(target.size()).to(device)

    # -------------------------------------------------------------------------
    # loop for other tasks
    # -------------------------------------------------------------------------
    #logits=[]
    for i in range(input.size()[1]):##
        # to maintain consistency in dimensions as torch.cat was throwing error
        ##in_data = torch.unsqueeze(input[i], 0)
        in_data = input[:,i,:]##(batch_size,in_feature)になるが、controllerのLSTMCell入力がちょうどこれなので問題なし
        out=ntm(in_data)


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # loop for NGram task
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------

    loss = criterion(out, target)
    losses.append(loss.item())
    loss.backward()
    # clips gradient in the range [-10,10]. Again there is a slight but
    # insignificant deviation from the paper where they are clipped to (-10,10)
    ##nn.utils.clip_grad_value_(ntm.parameters(), 10)
    nn.utils.clip_grad_norm_(ntm.parameters(), 50)
    optimizer.step()

    y_pred = torch.argmax(out.clone().detach(), dim=1)
    acc = accuracy_score(y_pred.cpu(), target.cpu())
    ##binary_output = out.clone()
    ##binary_output = binary_output.detach().cpu().apply_(lambda x: 0 if x < 0.5 else 1) ##

    # sequence prediction error is calculted in bits per sequence
    errors.append(acc.item())

    # ---logging---
    if iter % args.summarize_freq == 0:
        print('Iteration: %d\tLoss: %.2f\tError bits = acc per sequence: %.2f' %
              (iter, np.mean(losses), np.mean(errors)))
        #log_value('train_loss', np.mean(losses), iter)
        #log_value('bit_error_per_sequence', np.mean(errors), iter)
        scorestoring.store(data_config["savefileid_list"][0],iter,np.mean(errors))
        losses = []
        errors = []
        if np.mean(errors)>0.91 and (not a91_flag):
            slack = slackweb.Slack(url=config["slackurl"])
            slack.notify(text="over 91% !")
            a91_flag=True

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