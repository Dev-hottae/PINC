import pandas as pd
import os
from tokenizers import CharBPETokenizer, ByteLevelBPETokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer

### 전체 프로세스

## 00시 00분 scrapy crawler 작동


## 크롤링 종료 후 데이터 전처리
from Data_crawler.data_pre_processing import Pre
from topic_modeling.tokenizer import Tokenizer

path = r"Data_crawler/dataset"
# pre = Pre(keyword="삼성전자", path=path)
# pre.run_processing()
print("전처리 완료")

## TOPIC 모델링을 통한 1일 핵심 주제 키워드 추출
# data 토크나이저
# data load
print("데이터 로딩 중")
data_dir = 'Data_crawler/dataset/삼성전자_pred'
data_list = os.listdir(data_dir)
data = pd.DataFrame([])
for data_file in data_list:
    news = pd.read_json(data_dir+"/"+data_file, encoding="utf-8")
    data = data.append(news)
print("데이터 로딩 완료")
print("============")

print("데이터 토크나이징")
tok = Tokenizer(data=data, vocab_file=None)
# data 날짜 구간 조건
tok.date_condition(start_date="2005-01-01")
# # 데이터 맞춤법 교정 // 현재 속도 굉장히 느림
print("맞춤법 교정중")
tok.check_spell(data_trunc=20, max_char=500, dropna=True)
print("맞춤법 교정완료")
# mecab 을 활용한 불용어제거
print("불용어제거중")
tok.ex_stopword(True) ## 일단 mecab의 명사와 동사 추출
print("불용어제거완료")
# tokenizer train
print("토크나이저 사전 구축중")
tok.train_vocab(vocab_size=32000, min_freq=5, tokenizer=BertWordPieceTokenizer())
print("토크나이저 사전 구축완료")

# tokenize
print("토크나이징")
token_df = tok.tokenizer(tokenizer=BertWordPieceTokenizer(vocab_file='topic_modeling/huggingface_tokenizer_kor_32000-vocab.txt'))
print("토크나이징 완료")

# 데이터 1차 저장
token_df.to_json("after_tokenizing.json")




# LDA 토픽추출
print("LDA run...")
test = tok.get_lda(n=100)
print("LDA 완료")

## 추출된 키워드를 키반 GPT 알고리즘을 통해 자연어 생성
