#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Data setting

from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
xtrain, xtest, ytrain, ytest = train_test_split(iris.data, iris.target, test_size=0.5)

xtrain = torch.from_numpy(xtrain).type('torch.FloatTensor')
ytrain = torch.from_numpy(ytrain).type('torch.LongTensor')
xtest = torch.from_numpy(xtest).type('torch.FloatTensor')
ytest = torch.from_numpy(ytest).type('torch.LongTensor')

# Define model

class MyIris(nn.Module):
    def __init__(self):
        super(MyIris, self).__init__()
        self.l1=nn.Linear(4,6)
        self.l2=nn.Linear(6,3)
    def forward(self,x):
         h1 = torch.sigmoid(self.l1(x))
         h2 = self.l2(h1)
         return h2
     
# model generate, optimizer and criterion setting

model = MyIris()
optimizer = optim.SGD(model.parameters(),lr=0.1)
criterion = nn.CrossEntropyLoss()

# Learn

n = 75    ##  �f�[�^�̃T�C�Y
bs = 25   ##  �o�b�`�̃T�C�Y
model.train()
for i in range(1000):  ## �G�|�b�N���͂܂��K��
    idx = np.random.permutation(n)
    for j in range(0,n,bs):
        xtm = xtrain[idx[j:(j+bs) if (j+bs) < n else n]]
        ytm = ytrain[idx[j:(j+bs) if (j+bs) < n else n]]
        output = model(xtm)
        loss = criterion(output,ytm)
        print(i, j, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# torch.save(model.state_dict(),'myiris.model')     ## ���f���̕ۑ�
# model.load_state_dict(torch.load('myiris.model')) ## ���f���̌Ăяo��
    
# Test

model.eval()
with torch.no_grad():
    output1 = model(xtest)
    ans = torch.argmax(output1,1)
    print(((ytest == ans).sum().float() / len(ans)).item())



