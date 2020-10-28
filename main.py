import pandas as pd
import os
from tokenizers import CharBPETokenizer, ByteLevelBPETokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer

### 전체 프로세스

## 00시 00분 scrapy crawler 작동


## 크롤링 종료 후 데이터 전처리

from Data_crawler.data_pre_processing import Pre
from topic_modeling.ex_topic_news import Ex
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
# tok.check_spell(data_trunc=20, max_char=500, dropna=True)
print("맞춤법 교정완료")
# mecab 을 활용한 불용어제거
print("불용어제거중")
tok.ex_stopword(True, allow_type='n') ## 일단 mecab의 명사와 동사 추출
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



#### 성능개선 필요함 시간 오래걸림
# LDA 토픽추출
print("LDA run...")
topic_df = tok.get_lda(n=100, num_words=6)
print(topic_df)
topic_df.to_json("lda_100.json")
print("LDA 완료")



## stock_vol 데이터 로드



## topic 기준 뉴스데이터 선정
ex = Ex(topic_df, stock_vol)
ex.stop_topic(allow=3)
print("stop_topic after")
print(ex.topic_df)
ex.topic_count(tok.data[['date','text']])
print("topic count after")
print(ex.topic_df.head(50))
print(ex.topic_df.tail(50))

print("토픽 카운트 데이터 저장")
ex.topic_df.to_json("topic_count.json")


## 피크데이 정의
ex.peak_day_vol()
## 피크데이 뉴스 추출
ex.peak_day_news()

## 피크데이 토픽, 일자별 테이블 생성
ex.topic_news_count()

## 추출된 키워드를 키반 GPT 알고리즘을 통해 자연어 생성
