import json
from tqdm import tqdm
import numpy as np
import os,sys
import argparse

import torch
from torch import nn, optim

from ntm import NTM
from task_generator import CopyDataset, AssociativeDataset

import slackweb
sys.path.append("/work/handmade_utils/sotsuron_scores")
with __import__('importnb').Notebook(): 
    from score_store import ScoreStoring

# ==== Arguments ====
parser = argparse.ArgumentParser()
parser.add_argument('-task_json', type=str, default='configs/copy.json',
                    help='path to json file with task specific parameters')
parser.add_argument('-saved_model', default='model_copy.pt',
                    help='path to file with final model parameters')
parser.add_argument('-batch_size', type=int, default=1,
                    help='batch size of input sequence during training')
parser.add_argument('-num_steps', type=int, default=200000,
                    help='number of training steps')

parser.add_argument('-lr', type=float, default=1e-3,
                    help='learning rate for rmsprop optimizer')
parser.add_argument('-momentum', type=float, default=0.9,
                    help='momentum for rmsprop optimizer')
parser.add_argument('-alpha', type=float, default=0.95,
                    help='alpha for rmsprop optimizer')
parser.add_argument('-beta1', type=float, default=0.9,
                    help='beta1 constant for adam optimizer')
parser.add_argument('-beta2', type=float, default=0.999,
                    help='beta2 constant for adam optimizer')

args = parser.parse_args()

step_freq=1000
fileid="AssocR_Adam50"
config_file="./configs/config_all.json"
with open(config_file, "r") as fp:
    config = json.load(fp)
data_config=config["data"]


# ==== Create Dataset ====
# Copy Task
task_params = json.load(open(args.task_json))
#dataset = CopyDataset(task_params)
#Associative Recall Task
dataset = AssociativeDataset(task_params)

# ==== Create NTM ====
ntm = NTM(input_dim=task_params['seq_width'] + 2,
          output_dim=task_params['seq_width'],
          ctrl_dim=task_params['controller_size'],
          memory_units=task_params['memory_units'],
          memory_unit_size=task_params['memory_unit_size'],
          num_heads=task_params['num_heads'])

# ==== Training Settings ====
# Loss Function
criterion = nn.BCELoss()
#optimizer = optim.RMSprop(ntm.parameters(),
#                          lr=args.lr,
#                          alpha=args.alpha,
#                          momentum=args.momentum)

optimizer = optim.Adam(ntm.parameters(), lr=args.lr,
                        betas=(args.beta1, args.beta2))
sys.path.append("/work/handmade_utils/sotsuron_scores")
scorestoring =ScoreStoring(data_config,fileid_str_list=[fileid],rowAttr="steps",rowVal="bit error per seq")


# ==== Training ====
losses = []
errors = []

for step in tqdm(range(args.num_steps)):
    
    optimizer.zero_grad()
    ntm.reset()
    
    # Sample data
    data = dataset[step]
    inputs, target = data['input'], data['target']
    
    # Tensor to store outputs
    out = torch.zeros(target.size())
    
    # Process the inputs through NTM for memorization
    for i in range(inputs.size()[0]):
        # Forward passing all sequences for read
        ntm(inputs[i].unsqueeze(0))
        
    # Get the outputs from memory without real inputs
    zero_inputs = torch.zeros(inputs.size()[1]).unsqueeze(0) # dummy inputs
    for i in range(target.size()[0]):
        out[i] = ntm(zero_inputs)
    
    # Compute loss, backprop, and optimize
    loss = criterion(out, target)
    losses.append(loss.item())
    loss.backward()
    ##nn.utils.clip_grad_value_(ntm.parameters(), 10)
    nn.utils.clip_grad_norm_(ntm.parameters(), 50)
    optimizer.step()
    
    # Calculate binary outputs
    binary_output = out.clone()
    binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)
    
    # Sequence prediction error is calculted in bits per sequence
    error = torch.sum(torch.abs(binary_output - target))
    errors.append(error.item())
    
    # Print Stats
    if step % step_freq == 0:
        print('Step {} == Loss {:.3f} == Error {} bits per sequence'.format(step, np.mean(losses), np.mean(errors)))
        
        scorestoring.store(fileid,step,np.mean(errors))
        losses = []
        errors = []
        
slack = slackweb.Slack(url="https://hooks.slack.com/services/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")
slack.notify(text="training done !")
# Save model
torch.save(ntm.state_dict(), args.saved_model)
    
    
