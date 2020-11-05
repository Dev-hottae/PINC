import torch
import torch.nn.functional as F
import torch.optim as optim
import gluonnlp as nlp
import numpy as np
import pandas as pd
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm, tqdm_notebook
from transformers import AdamW, BertModel, BertConfig
from transformers.optimization import get_linear_schedule_with_warmup
from kobert.utils import get_tokenizer, download, tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from utils import BERT_Dataset_Train, BERTDataset, BERTClassifier, calc_accuracy

from Data_Analysis.utils import GPT_Dataset_Train

pytorch_kobert = {
    'url': 'https://kobert.blob.core.windows.net/models/kobert/pytorch/pytorch_kobert_2439f391a6.params',
    'fname': 'pytorch_kobert_2439f391a6.params',
    'chksum': '2439f391a6'
}

bert_config = {
    'attention_probs_dropout_prob': 0.1,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.1,
    'hidden_size': 768,
    'initializer_range': 0.02,
    'intermediate_size': 3072,
    'max_position_embeddings': 512,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'type_vocab_size': 2,
    'vocab_size': 8002
}

save_path = 'Data_Analysis/checkpoint'
ctx= 'cuda'
cachedir='~/kobert/'

print('모델 다운로드')

# download model
model_info = pytorch_kobert
model_path = download(model_info['url'],
                      model_info['fname'],
                      model_info['chksum'],
                      cachedir=cachedir)
# download vocab
vocab_info = tokenizer
vocab_path = download(vocab_info['url'],
                      vocab_info['fname'],
                      vocab_info['chksum'],
                      cachedir=cachedir)
#################################################################################################
print('BERT 모델 선언')

bertmodel = BertModel(config=BertConfig.from_dict(bert_config))
bertmodel.state_dict(torch.load(model_path))

print("GPU 디바이스 세팅")
device = torch.device(ctx)
bertmodel.to(device)
bertmodel.train()
vocab = nlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                               padding_token='[PAD]')

#################################################################################################
# 파라미터 세팅
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

max_len = 64
batch_size = 64
warmup_ratio = 0.1
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5
#################################################################################################
print("데이터를 준비중입니다.")
data_file_path = 'Data_crawler/dataset/삼성전자_pred/pre_삼성전자_연합인포맥스.json'

dataset_train, dataset_test = GPT_Dataset_Train(data_file_path)
data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)
#################################################################################################
num_epochs = 10

print("모델을 준비합니다. 에폭횟수 : {}".format(num_epochs))

model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
# 옵티마이져 설정
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
# 웜업 설정
t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
# 스케쥴러 설정
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=warmup_step,
                                            num_training_steps =t_total) # Warm-Up Step 추가

print('학습을 시작합니다.')
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    # 모델 학습 : model.train()을 지정해줘야 Fine Tunning이 이뤄짐
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    # 모델 평가 : model.eval()을 지정해야 Fine Tunning을 멈추고 평가를 시작함
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
    # 에폭마다 체크포인트 저장
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss' : loss.data.cpu().numpy()
                }, save_path+'Classification_KoBERT_checkpoint{}.tar'.format(num_epochs))