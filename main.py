import torch
from transformers import AutoTokenizer
from get_data import load_data,NERDataset,make_ner,get_ner
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from uer.roberta_crf import TeacherModel,TransformerCRF
#应该怎么写？？
##将id转成label。这个顺序不一样
label2id={
        'O': 0, 
    'B-address': 1, 
    'B-book': 3,
    'B-company': 5, 
    'B-game': 7,
    'B-government': 9, 
    'B-movie': 11, 
    'B-name': 13, 
    'B-organization': 15, 
    'B-position': 17, 
    'B-scene': 19, 
    'I-address': 2, 
    'I-book': 4, 
    'I-company': 6, 
    'I-game': 8, 
    'I-government': 10, 
    'I-movie': 12, 
    'I-name': 14, 
    'I-organization': 16, 
    'I-position': 18, 
    'I-scene': 20, 
    'S-address': 21, 
    'S-book': 22, 
    'S-company': 23, 
    'S-game': 24, 
    'S-government': 25, 
    'S-movie': 26, 
    'S-name': 27, 
    'S-organization': 28, 
    'S-position': 29, 
    'S-scene': 30, 
    '[PAD]': 31
    }
id2label = {_id: _label for _label, _id in list(label2id.items())}

# 配置日志记录
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def collate_fn(batch):
    input_ids_batch = []
    label_ids_batch = []
    mask_ids_batch=[]
    # Find the maximum sequence length in the batch
    # max_len = max(len(inputs[0]) for inputs in batch)
    max_len=64
    # Pad sequences to the maximum length
    for inputs in batch:
        input_ids, label_ids = inputs
        # Append padded sequences to the batch
        input_ids_batch.append(input_ids)
        label_ids_batch.append(label_ids)
        mask_ids_batch.append([1]*len(input_ids_batch))
    padded_sentences = []
    # Pad sequences to length 128 #和这个有关系
    padded_input_ids = pad_sequence([torch.tensor(ids+ [0] * (max_len - len(ids))) for ids in input_ids_batch], batch_first=True)
    # padded_attention_mask = pad_sequence([torch.tensor(ids) + [0] * (128 - len(ids)) for ids in attention_mask_batch], batch_first=True)
    padded_label_ids = pad_sequence([torch.tensor(ids + [31] * (max_len - len(ids)))  for ids in label_ids_batch], batch_first=True)
    padded_mask_ids=pad_sequence([torch.tensor(ids + [0] * (max_len - len(ids)))  for ids in mask_ids_batch], batch_first=True)
    return  padded_input_ids, padded_label_ids,padded_mask_ids
def evaluate(model, dataloader,ways):
    total_correct = 0
    total_predicted_positive = 0
    total_actual_positive = 0
    total_true_positive = 0
    predicted_list=[]
    label_list=[]
    for inputs, labels,padded_mask_ids in dataloader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        if ways=="teacher":
            logits,predicted= model(inputs)
        elif ways=="student":
            logits,loss,predicted= model(inputs,labels)
        
        # 计算预测值和标签的准确率、召回率、精确率和F1分数
        for pp in predicted:
            predicted_list.extend(pp)
        for row in labels:
            label_list.extend(row.tolist())
        total_correct=0
    total_correct_zero=0
    total_no_zero=0
    for p,l in zip(predicted_list,label_list):
        if p==l and l!=0:
            total_correct+=1
        if p==l and l==0:
            total_correct_zero+=1
        if p==l:
            total_actual_positive+=1
        if l!=0:
            total_no_zero+=1
    accuracy=total_actual_positive/len(predicted_list)
    no_zero_accuracy = total_correct / total_no_zero
    zero_accuracy=total_correct_zero/len(predicted_list)
    
    if ways=="teacher":
        print(f"teacher_model's Accuracy: {accuracy:.4f}")
        print(f"teacher_model's no_zero_Accuracy: {no_zero_accuracy:.4f}")
        print(f"teacher_model's zero_Accuracy: {zero_accuracy:.4f}")
    elif ways=="student":
        print(f"student_model's Accuracy: {accuracy:.4f}")
        print(f"student_model's no_zero_Accuracy: {no_zero_accuracy:.4f}")
        print(f"student_model's zero_Accuracy: {zero_accuracy:.4f}")
    return accuracy
# load_model()
train_data=load_data('/home/lhl/nlp/distill/dataset/train.txt')
valid_data=load_data('/home/lhl/nlp/distill/dataset/dev.txt')
# Load RoBERTa tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-cluener2020-chinese')
#student数值需要
num_classes = 32
num_layers = 2
hidden_size = 768
num_heads = 4
dropout = 0.1
alpha=0.5
epochs=100
seq="我家住在徐汇区虹漕路461号58号楼5楼"
model_name="/home/lhl/nlp/distill/model/student_model.pt"
get_ner(seq,tokenizer,model_name,id2label,"student")
teacher_model = TeacherModel()
print(teacher_model)
# for name, _ in teacher_model.named_parameters():
    # print(name)
student_model=TransformerCRF(num_classes, num_layers, hidden_size, num_heads, dropout)
# student_model.load_state_dict(torch.load("/home/lhl/nlp/distill/model/student_only_model.pt"))
#利用.logits
teacher_model.eval()
student_model.train()
# Create NERDataset instance
train_dataset = NERDataset(train_data, tokenizer,label2id)
valid_dataset = NERDataset(valid_data, tokenizer,label2id)

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,drop_last=True,collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,drop_last=True,collate_fn=collate_fn)

optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
#criterion = nn.CrossEntropyLoss()  # 用于计算子模型输出和真实标签的损失
ce_loss = nn.NLLLoss()#标签之间的差别
mse_loss = nn.MSELoss()#教师和学生之间的差别
cros_loss=torch.nn.CrossEntropyLoss()
best_accuracy=0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
student_model.to(device)
for name,params in teacher_model.named_parameters():
    params.requires_grad = False
# evaluate(student_model,valid_dataloader,"student")
# get_ner(seq,tokenizer,model_name,id2label)
outputs = []
for epoch in range(epochs):
    losses = []
    crf_list=[]
    student_list=[]
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        optimizer.zero_grad()
        student_model.train()
        input_ids, labels,masks = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        teacher_logits = teacher_model(input_ids=input_ids)
        student_logits, crf_loss, tags = student_model(input_ids=input_ids, labels=labels)
        # student_loss= student_model(input_ids=input_ids, labels=labels)
        student_loss=mse_loss(teacher_logits, student_logits)
        loss = alpha * crf_loss + (1 - alpha) * student_loss
        # loss=cros_loss(student_loss,labels)
        losses.append(loss.item())
        crf_list.append(crf_loss.item())
        student_list.append(student_loss.item())
        loss.backward()
        optimizer.step()
            # 记录训练结果到日志
        logging.info(f"Epoch {epoch} - loss: {np.mean(losses)}, crf_loss: {np.mean(crf_list)}, student_loss: {np.mean(student_list)}")
        progress_bar.set_postfix({"loss": np.mean(losses),"crf_loss": np.mean(crf_list),"student_loss": np.mean(student_list)})
    if epoch%5==0:
        # print("crf_loss:",crf_list)
        # print("logits_loss",student_loss)
        # make_ner(valid_dataloader,tokenizer,teacher_model,device,"teacher",id2label)
        # evaluate(teacher_model,valid_dataloader,"teacher")
        accuracy=evaluate(student_model,valid_dataloader,"student")
        if best_accuracy<accuracy:
            best_accuracy = accuracy
            # 保存模型
            torch.save(student_model.state_dict(), "/home/lhl/nlp/distill/model/student_only_model.pt")
logging.shutdown()
print("Training complete.")
