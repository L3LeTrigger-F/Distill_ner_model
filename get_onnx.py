#!/usr/bin/env python
# encoding: utf-8  
"""
@file: get_onnx.py
@description: 获得onnx模型和tfserving函数.数据输入格式：
"""
from transformers import AutoTokenizer,AutoModelForTokenClassification,TFAutoModelForTokenClassification
from uer.roberta_crf import TransformerCRF_eval,TransformerCRF
from onnx_tf.backend import prepare
from torch.nn.utils.rnn import pad_sequence
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
    alpha=0.7
    seq="北京朝阳区北苑华贸城"
    seq = list(seq)
    model=TransformerCRF_eval(num_classes, num_layers, hidden_size, num_heads, dropout)
    model.load_state_dict(torch.load("/home/lhl/nlp/Distill_ner_model/model/double_loss_no_crf_model.pt"))
    model.eval()
##save load
    max_len=64
    tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-cluener2020-chinese")
    token_ids = tokenizer.convert_tokens_to_ids(seq)
    token_ids=token_ids+[0]*(max_len - len(token_ids))
    token_ids=torch.tensor(token_ids)
    token_ids=token_ids.unsqueeze(0)
    torch.onnx.export(model,
                   token_ids,
                   "/home/lhl/nlp/Distill_ner_model/model/double_loss_no_crf_onnx.onnx",
                   opset_version=14,
                   training=torch.onnx.TrainingMode.EVAL, 
                   input_names=['input'],
                   output_names=['output'],  # the model's output names
                   dynamic_axes={'input': {0:'batch_size',1:'seq_length'},
                                 'output':{0:'batch_size',1:'seq_length'}
                                }
        )
# 加载模型
    model = onnx.load('/home/lhl/nlp/Distill_ner_model/model/double_loss_no_crf_onnx.onnx')
# 检查模型格式是否完整及正确
    onnx.checker.check_model(model)
# 获取输出层，包含层名称、维度信息
    input=model.graph.input
    output = model.graph.output
def inference_onnx_model(seq):
    model = onnx.load("/home/lhl/nlp/Distill_ner_model/model/double_loss_no_crf_onnx.onnx")
    from onnxruntime import InferenceSession
    provider = "CPUExecutionProvider"
    session = InferenceSession("/home/lhl/nlp/Distill_ner_model/model/double_loss_no_crf_onnx.onnx", providers=[provider])
    # seq="北京朝阳区北苑华贸城"
    seq = list(seq)
    max_len=64
    tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-cluener2020-chinese")
    token_ids = tokenizer.convert_tokens_to_ids(seq)
    token_ids=token_ids+[0]*(max_len - len(token_ids))
    token_ids=torch.tensor(token_ids)
    token_ids=token_ids.unsqueeze(0)#(1,10)
    token_ids=token_ids.detach().numpy()
    input_name='input'
    output_name='output'
    output = session.run([output_name], {"input":token_ids})
    output_data = output[0]

get_onnx()
inference_onnx_model("我家住在徐汇区虹漕路461号58号楼5楼")
# netron.start("/home/lhl/nlp/Distill_ner_model/model/double_loss_no_crf_onnx.onnx")
# netron.start("/home/lhl/nlp/distill/model/aa_try.onnx")
