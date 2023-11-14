#!/usr/bin/env python
# encoding: utf-8  
"""
@file: roberta_crf.py
@description: 模型结构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
import numpy as np
from transformers import BertTokenizer,AutoTokenizer,pipeline,AutoModelForTokenClassification
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-cluener2020-chinese')
        self.model = AutoModelForTokenClassification.from_pretrained('uer/roberta-base-finetuned-cluener2020-chinese',num_labels=32)
        labels = self.model.config.label2id 
        
    def forward(self,input_ids):
        #这个方法是取bert最后一层输出
        #logits = self.model(input_ids=input_ids).logits #(64,64,32)
        logits=self.model.bert(input_ids=input_ids, output_hidden_states=True)
        # predicted_labels = torch.argmax(logits, dim=2)[0].tolist()
        logits=logits.last_hidden_state#(64,64,768)
        # predicted_labels = torch.argmax(logits, dim=2).tolist()#64???
        # return logits,predicted_labels
        return logits

class TransformerCRF(nn.Module):
    def __init__(self,num_classes, num_layers, hidden_size, num_heads, dropout):
        super(TransformerCRF, self).__init__()
        self.vocab_size=10000
        self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size, dropout=dropout,batch_first=True),
            num_layers
        )
        self.hidden2tag = nn.Linear(hidden_size, num_classes)
        self.crf = CRF(num_tags=32,batch_first=True)

    def forward(self, input_ids,labels,if_crf):#(64,64) ##这里直接利用crf层计算损失
        '''带CRF层'''
        if if_crf==True:
            embedded = self.embedding(input_ids)#(64,64,32)
            logits = self.transformer_encoder(embedded)#(64,64,768)
            logits_sf=self.hidden2tag(logits)
            loss=self.crf(emissions=logits_sf,tags=labels)#直接用一个crf_score，作为标签。
            tags = self.crf.decode(logits_sf)
            return logits,-loss/10000,tags
            '''
        #直接蒸linear层
        else:
            embedded = self.embedding(input_ids)#(64,64,32)
            logits = self.transformer_encoder(embedded)#(64,64,768)
            logits_sf=self.hidden2tag(logits)
            predicted_label = torch.argmax(logits_sf,dim=-1)
            # loss=self.crf(emissions=logits_sf,tags=labels)#直接用一个crf_score，作为标签。
            # tags = self.crf.decode(logits_sf)
            return logits_sf,predicted_label
            '''
        else:
            embedded = self.embedding(input_ids)#(64,64,32)
            logits = self.transformer_encoder(embedded)#(64,64,768)
            logits_sf=self.hidden2tag(logits)
            predicted_label = torch.argmax(logits_sf,dim=-1)
            # loss=self.crf(emissions=logits_sf,tags=labels)#直接用一个crf_score，作为标签。
            # tags = self.crf.decode(logits_sf)
            return logits,logits_sf,predicted_label

class TransformerCRF_eval(nn.Module):
    def __init__(self,num_classes, num_layers, hidden_size, num_heads, dropout):
        super(TransformerCRF_eval, self).__init__()
        self.vocab_size=10000
        self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size, dropout=dropout,batch_first=True),
            num_layers
        )
        self.hidden2tag = nn.Linear(hidden_size, num_classes)
        self.crf = CRF(num_tags=32,batch_first=True)

    def forward(self, input_ids):#(64,64) ##这里直接利用crf层计算损失
        '''
        ##带CRF层
        embedded = self.embedding(input_ids)#(64,64,32)
        # embedded = embedded.permute(1, 0, 2)  # 转置输入维度顺序(64,32,64)
        logits = self.transformer_encoder(embedded)#(64,64,768)
        # logits_sf=logits.transpose(0, 1)
        logits_sf=self.hidden2tag(logits)
        # labels=labels.transpose(0, 1)#(64,64)
        tags = self.crf.decode(logits_sf)
        # return np.array(tags)
        return torch.tensor(tags)
        '''
        #去掉crf层的方法
        embedded = self.embedding(input_ids)#(64,64,32)
        logits = self.transformer_encoder(embedded)#(64,64,768)
        logits_sf=self.hidden2tag(logits)
        predicted_label = torch.argmax(logits_sf,dim=-1)
        return logits,logits_sf,predicted_label
