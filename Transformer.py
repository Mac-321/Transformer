# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:52:35 2024
@author: Ben
"""
import numpy as np
import math
import torch
from torch import nn
import transformers
from transformers import AutoTokenizer ,BertTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import xlwings as xw
from openpyxl import load_workbook
import torchtext
from torchtext import datasets
import csv
import  os
# pos - position of  word in the input sentence 0 <= k <= L/2
# dModel - dimension of the output embedding
# # i - the number of the word
class Transformer(nn.Module):
    def __init__(self, dModel,numEncode,numDecode,h,headSize,vocab_size):
        super().__init__()
        self.dModel= dModel
        self.vocab = vocab_size
        self.numHeads = h
        self.headSize = headSize
        self.Ne = numEncode
        self.Nx = numDecode
        self.encoderBlock = nn.ModuleList(Encoder(self.numHeads,self.dModel,self.headSize) for i in range(self.Ne))
        self.decoderBlock = nn.ModuleList(Decoder(self.numHeads,self.dModel,self.headSize) for i in range(self.Nx))
        self.out = nn.Linear(self.dModel,self.dModel)
        self.soft = nn.Softmax()
        
    def getDmodel(self):
        return self.dModel
    
    def getVocab(self):
        return self.vocab
    
    def forward(self,x):
        print("hello")
        print("start of transformer model",x.size())
        logits = torch.empty([0])
        #encoderOut = torch.empty([0])
        encoderOutK = torch.empty([0])
        encoderOutV =torch.empty([0])
        for i in (self.encoderBlock):
            x,K,V = i(x)
        #print("hell0obgfgfer",x.size())
        for i in (self.decoderBlock):
            j = i(x,K,V)
        logits = j
        logits = self.out(logits)
        logits = self.soft(logits)
        return logits

class Decoder(nn.Module):
    def __init__(self,h,dModel,headSize):
        super().__init__()
        self.numOfHeads = h
        self.dModel =dModel
        self.crossHMask = None#torch.full([1,1],-math.inf).triu()
        #print("Helllllllo",self.crossHMask)
        self.attentionMask = torch.zeros([1,1])
        self.headSize = headSize
        self.lNorm = nn.LayerNorm([2,self.dModel])
        self.multiHeadAttention = nn.ModuleList(Attention(self.dModel,self.headSize,self.attentionMask,self.numOfHeads ) for i in range(self.numOfHeads))
        self.crossHeadAttention = nn.ModuleList(CrossAttention(self.dModel,self.headSize,self.crossHMask,self.numOfHeads ) for i in range(self.numOfHeads))
        self.MLP = nn.Sequential(nn.Linear(self.dModel,4*self.dModel),
                                 nn.ReLU(),
                                 nn.Linear(4*self.dModel,self.dModel))
        self.linOut = nn.Sequential(nn.Linear(self.dModel,self.dModel),
                                    nn.Softmax())

    def forward(self,x,encodeK,encodeV):
        out = torch.empty([0])
        out2 =torch.empty([0])
        for i in self.multiHeadAttention:
            outlmo,_,_ = i(x)
            print("ello",outlmo.size())
            out = torch.cat([out,outlmo],dim=1)
        out1 = self.lNorm(x+out)
        for i in self.crossHeadAttention:
            temp = i(out,encodeK,encodeV)
            out2 = torch.concat([out2,temp],dim=1)
        out3 = self.lNorm(out2+out1)
        logits = self.MLP(out3)
        logits = self.lNorm(logits+out3)
        logits = self.linOut(logits)

        return logits
    
class MoE(nn.Module):

    def __init__(self,h,dModel,headSize,numOfE,numT,c):
     super().__init__()
     self.numOfHeads = h
     self.dModel =dModel
     self.e =numOfE
     self.c =c
     self.k = numT*self.c/self.e
     self.lin = nn.Linear(self.dModel, self.e)
     
    def forward(self,x):
        #print("At start encoder new input token is no2",x.size())
        out = self.lin(x)
        out = nn.Softmax(out)
        out =torch.topk(out.transpose(),self.k)
        
class Encoder(nn.Module):

    def __init__(self,h,dModel,headSize):
     super().__init__()
     self.numOfHeads = h
     self.dModel =dModel
     self.attentionMask = None#torch.zeros([1,1])
     self.headSize = headSize
     self.lNorm = nn.LayerNorm([2,self.dModel])
     self.multiHeadAttention = nn.ModuleList(Attention(self.dModel,self.headSize,self.attentionMask,self.numOfHeads) for i in range(self.numOfHeads))
     self.dimensionP = self.dModel/self.numOfHeads
     self.linear = nn.Linear(int(self.numOfHeads*self.dimensionP),self.dModel)
     self.MLP = nn.Sequential(nn.Linear(self.dModel,4*self.dModel),
                              nn.ReLU(),
                              nn.Linear(4*self.dModel,self.dModel))
    def forward(self,x):
        #print("At start encoder new input token is no2",x.size())
        out = torch.empty([0])
        K= torch.empty([0])
        V = torch.empty([0])
        for i in self.multiHeadAttention:
            outE ,KE,VE =i(x)
            #print(outE.size())
            out = torch.cat([out,outE],dim=1)
            K = torch.cat([K,KE],dim=1)
            V = torch.cat([V,VE],dim=1)
        print(V.size(),K.size(),out.size())
        #print(Vout.size())
        out = self.linear(out)
        K =self.linear(K)
        V = self.linear(V)
        out1 = self.lNorm(x+out)
        out2 = self.MLP(out1)
        logits = self.lNorm(out1+out2)
       # print("out of encoder",logits.size())
        #print("fuckfuck",K.size())
        return logits,K,V


class EmbedOrDecode(nn.Module):
    def __init__(self,dModel,vocab):
        super().__init__()
        self.outDim =dModel
        self.vocab = vocab
        self.embedTable = nn.Embedding(30522,self.outDim)
        
    def posEncode(self,inputDim):
        i= np.arange(inputDim)[:,np.newaxis]
        pos=np.arange(self.outDim)[np.newaxis]
        i = torch.tensor(i)
        pos = torch.tensor(pos)
        exponent =2*i/self.outDim
        x = np.radians(pos/(10000**(exponent)))
        y= torch.sin(x)
        y1= torch.cos(x)
        out = torch.concat([y,y1])
        return out
    
    def encode(self,x):
        print(len(x))
        pos = self.posEncode(len(x))
        print(pos.size())
        for i,j in enumerate(x):
         print(j)
         x = self.embedTable(torch.LongTensor([1,j]))
        
        print(x.size())
        x = x +pos[0]
        
        return x
        
class Attention(nn.Module):
    def __init__(self,dModel,headSize,mask,numOfHeads):
        super().__init__()
        self.numHeads=numOfHeads
        self.Soft = nn.Softmax()
        self.mask = mask
        self.dModel = dModel
        self.headSize= int(self.dModel/self.numHeads)
        self.linearQ = nn.Linear(self.dModel,self.headSize)
        self.linearK = nn.Linear(self.dModel,self.headSize)
        self.linearV = nn.Linear(self.dModel,self.headSize)
        self.dimensionP = self.dModel/self.numHeads

    def forward(self,data):
        print("data size",data.size())
        Q = self.linearQ(data)
        K = self.linearK(data)
        V = self.linearV(data)
        print(Q.size())
        #only transpose data dimensions not batch dimensions
        logits = torch.matmul(Q,K.transpose(0,1))*1/(math.sqrt(self.dimensionP))
        print("hell",logits.size())
        #Sort mask out
        if self.mask != None:
            logits +=self.mask
        #logits
        logits1 = self.Soft(logits)
        logits = torch.matmul(logits1,V)
        print("logits size atten",logits.size())
        #print("size atten",K.size())
        return logits, K,V

class CrossAttention(Attention):
    def __init(self,dModel,headSize):
        super().__init__()

    def forward(self,data,encodeK,encodeQ):
       # print("datasss",encodeK.size())
        Q = self.linearQ(encodeQ)
        K = self.linearK(encodeK)
        V = self.linearV(data)
        #print("heyy",V.size(),Q.size(),K.size())
       # K =torch.transpose(K,0,1)
        logits= torch.matmul(Q,K.transpose(0,1))*1/(math.sqrt(self.dimensionP))
        #Sort mask out
        if self.mask != None:
            logits +=self.mask
        #logits
        logits = self.Soft(logits)
        logits = torch.matmul(logits,V)
        return logits

def trainNN(model,tokenizedQs,tokenizedAs,optim,loss):
    model.train()
    data= tokenizedQs
    ans = tokenizedAs
    print(model.getDmodel())    
    print(model.getVocab()) 
    embedder = EmbedOrDecode(dModel,30522)
    #data = embedder.encode(data)
    data = embedder.encode(data)
    ans = embedder.encode(ans)
    print(data.size())
    optim.zero_grad()
    pred = model(data)
    print(pred.size())
    Loss = loss(pred,ans)
    Loss.backward()
    optim.step()

def tokenizeData(data,tokenizer):
    #print(data)
    tokenizedD = []#tokenizer(data,padding = True, truncation=True,return_tensors='pt')
    #print(tokenizedD.size())
    for i in data:
        #for j in  i.split():
        #print("Data:",j)
        w= tokenizer(i)
        w = w['input_ids']
       # w =torch.tensor(w)
        #print(w)
        #tokenizedD.append(w)
        #print(len(tokenizedD))
    tokenizedD = w
    #print(tokenizedD.size())
    return tokenizedD

file_name = 'medquad.csv'
df = pd.read_csv(file_name)
#print(df)
df.head()
df = df.fillna('')
questions ,response = df['question'].values,df['answer'].values
#(questions)
#tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = BertTokenizer.vocab_size
tokenizedQs = tokenizeData(questions,tokenizer)
tokenizedAs = tokenizeData(response,tokenizer)
#print(tokenizedQs)
dModel =512
numEncode=6
numDecode =6
h =8
headSize = dModel/h
learningRate= 5e-5
model =Transformer(dModel, numEncode, numDecode,h,headSize,vocab_size)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
loss = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(),lr =learningRate)
#checkpoint = torch.load('transformer.pth')
#model.load_state_dict(checkpoint['transformer_state_dict'])
#optim.load_state_dict(checkpoint['optim_state_dict'])
print("num of parameters:",params)
input =["hello there!"]
BSize =64
#data = DataLoader(dataset, batch_size=BSize,drop_last=True)
for i in range(100):
    trainNN(model,tokenizedQs,tokenizedAs,optim,loss)
    #print(model(tokenizedQs))
torch.save({
            'transformer_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            },'transformer.pth')