# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U2GEP0LTB7YedflFPvvlYAPYNZ5ApnqD
"""

!pip3 install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

!pip install transformers
!pip install onnx
!pip install netron

# Commented out IPython magic to ensure Python compatibility.
!pip uninstall pytorch-crf
# %cd /content/pytorch-crf-jit-scriptable/pytorch-crf-jit-scriptable
!python setup.py install

!pip install netron

!pip install pytorch-crf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
import numpy as np
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
        embedded = self.embedding(input_ids)#(64,64,32)
        logits = self.transformer_encoder(embedded)#(64,64,768)
        logits_sf=self.hidden2tag(logits)
        predicted_label = torch.argmax(logits_sf,dim=-1)
        return logits,logits_sf,predicted_label

from google.colab import drive
drive.mount('/content/drive')
import os

!unzip "/content/pytorch-crf-jit-scriptable.zip" -d "/content/pytorch-crf-jit-scriptable/"



!git clone https://github.com/kmkurn/pytorch-crf.git

from transformers import AutoTokenizer,AutoModelForTokenClassification
# from uer.roberta_crf import TransformerCRF_eval,TransformerCRF
import torch
import torch.nn as nn
import numpy as np
import onnx
import netron
def get_onnx():
    num_classes = 32
    num_layers = 2
    hidden_size = 768
    num_heads = 4
    dropout = 0.1
    alpha=0.5
    text_tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-cluener2020-chinese")
    model=TransformerCRF_eval(num_classes, num_layers, hidden_size, num_heads, dropout)
    # model=TransformerCRF(num_classes, num_layers, hidden_size, num_heads, dropout)
    # model = torch.load('/home/lhl/nlp/distill/model/student_model.pt')
    model.load_state_dict(torch.load("/content/double_loss_no_crf_model.pt",map_location=torch.device('cpu')))
    # torch.save(model.state_dict(),"/home/lhl/nlp/distill/model/save.pt")

    # model = torch.load("/home/lhl/nlp/distill/model/save.pt")
    model.eval()
    # print(model)
    # torch.save(tests, '/home/lhl/nlp/distill/examples/model_path/text_encoder.pt')
    # tests=torch.load("/home/lhl/nlp/distill/examples/model_path/text_encoder.pt")
##save load
    text1 = text_tokenizer(["北京朝阳区北苑华贸城"], return_tensors='pt', padding=True)["input_ids"]
    # text1 = text_tokenizer(["北京朝阳区北苑华贸城"])
    # '''从这里开始试
    torch.onnx.export(model,
                   text1,
                   "/content/ner_distill_model.onnx",
                  #  "/home/lhl/nlp/distill/model/aa_try.onnx",
                   opset_version=11,
                   input_names=["input"],
                   output_names=["output"],  # the model's output names
                   dynamic_axes={'input': {0:'batch_size',1:'seq_length'}}
        )
# 加载模型
    # model = onnx.load('/home/lhl/nlp/distill/model/ner_distill_model.onnx')
    model = onnx.load('/content/ner_distill_model.onnx')
# 检查模型格式是否完整及正确
    onnx.checker.check_model(model)
    # print(model.graph)
# 获取输出层，包含层名称、维度信息
    input=model.graph.input
    output = model.graph.output
    print(input)
    print(output)

get_onnx()

!pip install onnxruntime

import numpy as np
from torch.nn.utils.rnn import pad_sequence
def inference_model():
#     model = onnx.load("/home/lhl/nlp/distill/model/ner_distill_model.onnx")
    from onnxruntime import InferenceSession
    provider = "CPUExecutionProvider"
    session = InferenceSession("/content/ner_distill_model.onnx", providers=[provider])
    seq=["北","京","朝","阳","区","北","苑","华","贸","城"]
    tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-cluener2020-chinese")
    token_ids = tokenizer.convert_tokens_to_ids(seq)
    # print(token_ids)
    # token_ids=torch.tensor(token_ids)
    # token_ids=token_ids.unsqueeze(0)
    token_ids=[token_ids]
    padded_token_ids=pad_sequence([torch.tensor(ids+[0]*(64-len(ids))) for ids in token_ids],batch_first=True)
    padded_token_ids=padded_token_ids.unsqueeze(0)
    padded_token_ids=np.array(padded_token_ids)
    print(padded_token_ids.shape)
    # token_ids=token_ids.detach().numpy()
    # aa=session.get_inputs()[0].name
    # print(aa)
    # output_tensors = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(input_name)
    print(output_name)
    # input_name='input'
    # output_name='output'
    a= session.run([output_name], {input_name:token_ids})
    # output_data = output[0]
    print(a)

inference_model()
# netron.start("/home/lhl/nlp/distill/model/ner_distill_model.onnx")
# netron.start("/home/lhl/nlp/distill/model/aa_try.onnx")
