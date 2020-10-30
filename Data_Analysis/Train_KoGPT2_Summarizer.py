import torch
from torch.utils.data import DataLoader, Dataset
from gluonnlp.data import SentencepieceTokenizer 
import gluonnlp as nlp
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from kogpt2.utils import download, tokenizer, get_tokenizer
from kogpt2.pytorch_kogpt2 import GPT2Config, GPT2LMHeadModel
from utils import GPT_Dataset_Train, GPTDataset

pytorch_kogpt2 = {
    'url':
    'https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params',
    'fname': 'pytorch_kogpt2_676e9bcfa7.params',
    'chksum': '676e9bcfa7'
}
kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000,
    "output_past": None
}

ctx= 'cuda'#'cuda' #'cpu' #학습 Device CPU or GPU. colab의 경우 GPU 사용
cachedir='~/kogpt2/' # KoGPT-2 모델 다운로드 경로
save_path = 'Data_Analysis/checkpoint'

print("모델 다운로드")

# download model
model_info = pytorch_kogpt2
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
##########################################################################################
print("GPT2 모델선언")
# KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
kogpt2model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=None,
                                              config=GPT2Config.from_dict(kogpt2_config),
                                              state_dict=torch.load(model_path))

device = torch.device(ctx)
kogpt2model.to(device)
# Fine Tunning을 위해 train 선언
kogpt2model.train()
# 단어 뭉치 가져오기
vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                     mask_token=None,
                                                     sep_token=None,
                                                     cls_token=None,
                                                     unknown_token='<unk>',
                                                     padding_token='<pad>',
                                                     bos_token='<s>',
                                                     eos_token='</s>')
##########################################################################################
tok_path = get_tokenizer()
vocab = vocab_b_obj
sentencepieceTokenizer = SentencepieceTokenizer(tok_path)

print("데이터 로드")

data_file_path = 'Data_crawler/dataset/삼성전자_pred/pre_삼성전자_연합인포맥스.json'

news_data = GPT_Dataset_Train(data_file_path)
news_dataset = GPTDataset(news_data, vocab, sentencepieceTokenizer)  # Torch DataLoader 형태 맞춰주는 Dataset 설정
news_data_loader = DataLoader(news_dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=0)

# 파라미터 설정
learning_rate = 1e-5
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(kogpt2model.parameters(), lr=learning_rate)

for epoch in range(10):
    count = 0
    avg_loss = (0.0, 0.0)
    for data in tqdm(news_data_loader):
        optimizer.zero_grad()
        # Data에 Torch 스택
        data = torch.stack(data)
        data = data.transpose(1,0)
        # 데이터와 모델에 GPU 설정
        data = data.to(device)
        kogpt2model = kogpt2model.to(device)
        # 결과값
        outputs = kogpt2model(data, labels=data)
        loss, logits = outputs[:2]
        loss = loss.to(device)
        loss.backward()
        avg_loss = (avg_loss[0] * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)
        optimizer.step()
        count+=1

        if count % 1000 == 0:
            print('epoch no.{0} train no.{1}  loss = {2:.5f} avg_loss = {3:.5f}' . format(epoch+1, count, loss, avg_loss[0] / avg_loss[1]))

    # 에폭 마다 모델 저장
    torch.save({
        'model_state_dict': kogpt2model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss' : loss
        }, save_path+'Summarizer_KoGPT2_checkpoint{}.tar'.format(epoch+1))