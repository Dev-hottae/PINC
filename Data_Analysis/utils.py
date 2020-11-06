import pandas as pd
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset
from kogpt2.utils import download, tokenizer, get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
import gluonnlp as nlp


def sentencePieceTokenizer():
    tok_path = get_tokenizer()
    sentencepieceTokenizer = SentencepieceTokenizer(tok_path)

    return sentencepieceTokenizer

def koGPT2Vocab():
    cachedir = '~/kogpt2/'
    # download vocab
    vocab_info = tokenizer
    vocab_path = download(vocab_info['url'],
                        vocab_info['fname'],
                        vocab_info['chksum'],
                        cachedir=cachedir)

    koGPT2_vocab = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                                mask_token=None,
                                                                sep_token=None,
                                                                cls_token=None,
                                                                unknown_token='<unk>',
                                                                padding_token='<pad>',
                                                                bos_token='<s>',
                                                                eos_token='</s>')
    return koGPT2_vocab

def GPT_Dataset_Train(path):
    data_path = path
    news_data = pd.read_json(data_path, encoding='utf-8')

    news_data['text'] = news_data['text'].apply(lambda x: ". ".join(x[:10]))

    dataset_train = []
    for i in tqdm(range(len(news_data))):
        dataset_train.append(news_data['Topic_keyword'][i] + ". " + news_data['text'][i])
    return dataset_train

class GPTDataset(Dataset):
    def __init__(self, data_file, vocab, tokenizer):
        self.data =[]
        self.vocab = vocab
        self.tokenizer = tokenizer

        for data in data_file:
            tokenized_line = self.tokenizer(data)
            if len(tokenized_line) <= 1020: # 문장 총길이 1022로 제한
                index_of_words = [vocab.bos_token] + tokenized_line + [vocab.padding_token] * (1020 - len(tokenized_line)) + [vocab.eos_token]
                self.data.append(vocab(index_of_words))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        item = self.data[index]
        return item
###############################################################################################################################################
def BERT_Dataset_Train(path):
    data_path = path
    news_data = pd.read_json(data_path, encoding='utf-8').reset_index(drop=True)
    news_data = news_data[['date', 'text', 'label']]

    # 학습, 테스트 데이터 분리
    train_data = news_data.sample(frac=0.8, random_state=2020)
    train_data = train_data.dropna()
    train_data = train_data.reset_index()
    train_data = train_data.drop(['index'], axis=1)
    test_data = news_data.drop(train_data.index)
    test_data = test_data.dropna()
    test_data = test_data.reset_index()
    test_data = test_data.drop(['index'], axis=1)
    # 라벨 비율 체크
    count_0 = 0
    count_1 = 0
    for i in train_data['label']:
        if i == 0:
            count_0 += 1
        else:
            count_1 += 1
    print("라벨 비율 : {}".format(count_0 / (count_0 + count_1)))
    # 데이터셋 구축
    dataset_train = [] # 라벨은 0부터 순서대로 입력해야함
    dataset_test = []
    for i in tqdm(range(len(train_data))):
        dataset_train.append([train_data['text'][i], int(train_data['label'][i])]) # 해당 리스트 형태를 맞춰야 학습 가능
    for i in tqdm(range(len(test_data))):
        dataset_test.append([test_data['text'][i], int(test_data['label'][i])])

    return dataset_train, dataset_test
    
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2, # 해당 부분 파라미터 조정으로 다중 분류 가능
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

def calc_accuracy(X,Y): # 정확도 계산 함수
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
