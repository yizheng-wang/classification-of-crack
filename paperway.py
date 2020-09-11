#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 23:47:01 2020

@author: wangyizheng
"""

from torch import nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data
import numpy as np
class VGG(nn.Module):
    def __init__(self,num_class=1000):
        super(VGG, self).__init__()

        self.features = nn.Sequential(nn.Conv2d(3, 48, 4, padding=2),
                                      nn.ReLU(inplace = True),
                                      nn.MaxPool2d(2,2),
                                      nn.Conv2d(48, 48, 5, padding=2),
                                      nn.ReLU(inplace = True),
                                      nn.MaxPool2d(2,2),
                                      nn.Conv2d(48, 48, 3, padding=1),
                                      nn.ReLU(inplace = True),
                                      nn.MaxPool2d(2,2),
                                      nn.Conv2d(48, 48, 4, padding=2),
                                      nn.ReLU(inplace = True),
                                      nn.MaxPool2d(2,2))
        self.classifier = nn.Sequential(
            nn.Linear(48*14*14, 400),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(400, num_class)                        
                       )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        m = nn.Softmax(dim = 1)
        x = m(x)
        return x

import torch
from time import time
import cv2
import glob
vgg = VGG(2).cuda()
trainpositive = glob.glob('./ConcreteCrackImagesforClassification/trainpositive/*.jpg')
trainnegative = glob.glob('./ConcreteCrackImagesforClassification/trainnegative/*.jpg')
criterion = nn.CrossEntropyLoss()

# create a initial tensor
result = []
n=0
start = time()

Batch_size = 20





if __name__ == '__main__':
    for epoch in range(20):
        for everytrainbatch in range(14):
            x = torch.zeros(2000, 3, 227, 227).cuda()
            y = torch.zeros(2000).cuda()
            for i in range(1000):
                image = cv2.imread(trainpositive[everytrainbatch * 1000 + i])
                image = torch.from_numpy(image)
                image = image.permute(2, 0, 1).reshape(1, 3, 227, 227).cuda()
                image = image.type_as(vgg.features[0].weight)
                x[i] = image
                y[i] = torch.tensor([1])
    
                image = cv2.imread(trainnegative[everytrainbatch * 1000 + i])
                image = torch.from_numpy(image)
                image = image.permute(2, 0, 1).reshape(1, 3, 227, 227).cuda()
                image = image.type_as(vgg.features[0].weight)
                x[i + 1000] = image
                y[i + 1000] = torch.tensor([0])
                n += 1
                print(f'data process is {n}')
            print('train is done{everytrainbatch},total is 14')

            torch_dataset = Data.TensorDataset(x, y)

            loader = Data.DataLoader(
                dataset=torch_dataset,      # 数据，封装进Data.TensorDataset()类的数据
                batch_size=Batch_size,      # 每块的大小
                shuffle=True,               # 要不要打乱数据 (打乱比较好)
                num_workers=0,              # 多进程（multiprocess）来读数据
                )
            for  step,(batch_x, batch_y) in enumerate(loader):
                output = vgg(batch_x).cuda()            
                loss_nn = criterion(output.float(), batch_y.long())
                loss_nn.backward()
                optimizer = optim.SGD(params = vgg.parameters(), lr = 0.0001)
                optimizer.step()
                optimizer.zero_grad()  
                print(loss_nn)







'''for i in trainnegative:
    image = cv2.imread(i)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).reshape(1, 3, 227, 227).cuda()
    image = image.type_as(vgg.features[0].weight)
    output = vgg(image).cuda()
    label = torch.tensor([[0, 1]]).cuda()
    loss_nn = criterion(output.float(), label.float())
    loss_nn.backward()
    optimizer = optim.SGD(params = vgg.parameters(), lr = 0.001)
    optimizer.step()
    optimizer.zero_grad()
    n += 1    
    print(f'iter number is {n},and loss is {loss_nn}')
print('trainnegative is done')'''

# val
valpositive = glob.glob('./ConcreteCrackImagesforClassification/valpositive/*.jpg')
valnegative = glob.glob('./ConcreteCrackImagesforClassification/valnegative/*.jpg')
resultpositive = []
resultnegative = []
for i in valpositive:
    image = cv2.imread(i)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).reshape(1, 3, 227, 227).cuda()
    image = image.type_as(vgg.features[0].weight)
    output = vgg(image).cuda()
    resultpositive.append(output.max(1).indices)
    n += 1    
    print(n)
print('valpositive is done')
for i in valnegative:
    image = cv2.imread(i)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).reshape(1, 3, 227, 227).cuda()
    image = image.type_as(vgg.features[0].weight)
    output = vgg(image).cuda()
    resultnegative.append(output.max(1).indices)
    n += 1    
    print(n)
print('valnegative is done')
# In[0]
testpositive = glob.glob('./ConcreteCrackImagesforClassification/testpositive/*.jpg')
testnegative = glob.glob('./ConcreteCrackImagesforClassification/testnegative/*.jpg')
resultpositive = []
resultnegative = []
for i in testpositive:
    image = cv2.imread(i)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).reshape(1, 3, 227, 227).cuda()
    image = image.type_as(vgg.features[0].weight)
    output = vgg(image).cuda()
    resultpositive.append(output.max(1).indices)
    n += 1    
    print(n)
print('testpositive is done')
for i in testnegative:
    image = cv2.imread(i)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).reshape(1, 3, 227, 227).cuda()
    image = image.type_as(vgg.features[0].weight)
    output = vgg(image).cuda()
    resultnegative.append(output.max(1).indices)
    n += 1    
    print(n)
print('testnegative is done')
# In[1] evalute 
resultpositive = np.array(resultpositive)
resultnegative = np.array(resultnegative)
accpositive = resultpositive.sum().item()/len(resultpositive)
accnegative = (len(resultnegative)-resultnegative.sum().item())/len(resultnegative)
print(f'accuracy of positive is {accpositive}')
print(f'accuracy of negative is {accnegative}')
print(f'accuracy of total is {(accpositive+accnegative)/2}')
wrong_positive = [idx for idx, value in enumerate(resultpositive) if value == 0]
wrong_negative = [idx for idx, value in enumerate(resultnegative) if value == 1]
wrong_positive_filedir = [testpositive[i] for i in wrong_positive]
wrong_negative_filedir = [testnegative[i] for i in wrong_negative]

# In[0]
'''from PIL import Image
import os
for i in wrong_positive_filedir:
    picname = os.path.basename(i)
    pic = Image.open(i)
    pic.save(f'./ConcreteCrackImagesforClassification/wrong_positive/{picname}')
for i in wrong_negative_filedir:
    picname = os.path.basename(i)
    pic = Image.open(i)
    pic.save(f'./ConcreteCrackImagesforClassification/wrong_negative/{picname}')'''

