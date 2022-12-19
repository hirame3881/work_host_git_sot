#!/usr/bin/python
# -*- coding: sjis -*-

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import pickle

argvs = sys.argv
argc = len(argvs)

###################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##################

with open('dic.pkl','br') as f:
    dic = pickle.load(f)

labels = {'����': 0, '����': 1, '�`�e��': 2,
     '������': 3, '�⏕�L��': 4, '����': 5, '�㖼��': 6,
     '�ڔ���': 7, '����': 8, '�`��': 9, '�L��': 10,
     '�A�̎�': 11, '�ړ���': 12, '�ڑ���': 13,
     '������': 14, '��': 15}

# Data setting

class MyDataset(Dataset):
    def __init__(self, xdata, ydata):
        self.data = xdata
        self.label = ydata
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return (x,y)

def my_collate_fn(batch):
    images, targets= list(zip(*batch))
    xs = list(images)
    ys = list(targets)
    return xs, ys

with open('xtest.pkl','br') as fr:
    xdata = pickle.load(fr)

with open('ytest.pkl','br') as fr:
    ydata = pickle.load(fr)

batch_size = 200
dataset = MyDataset(xdata,ydata)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

# Define model

class MyLSTM(nn.Module):
    def __init__(self, vocsize, posn, hdim):
        super(MyLSTM, self).__init__()
        self.embd = nn.Embedding(vocsize, hdim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=hdim, hidden_size=hdim, batch_first=True, num_layers=2)
        self.ln   = nn.Linear(hdim, posn)
    def forward(self, x):
        x = self.embd(x)
        lo, (hn, cn) = self.lstm(x)
        out = self.ln(lo)
        return out

# model generate, optimizer and criterion setting

net = MyLSTM(len(dic)+1, len(labels), 100).to(device)
net.load_state_dict(torch.load(argvs[1]))

# Eval

real_data_num = 0
net.eval()
with torch.no_grad():
    ok = 0
    for xs, ys in dataloader:
        xs1, ys1 = [], []
        for k in range(len(xs)):
            real_data_num += len(xs[k])
            tid = xs[k]
            xs1.append(torch.LongTensor(tid))
            tid = ys[k]
            ys1.append(torch.LongTensor(tid))
        xs1 = pad_sequence(xs1, batch_first=True).to(device)
        ys1 = pad_sequence(ys1, batch_first=True, padding_value=-1.0).to(device)
        output = net(xs1)
        ans = torch.argmax(output,dim=2)
        ok += torch.sum(ans == ys1).item()
print(ok, real_data_num, ok/real_data_num)
