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
tok = Tokenizer(data=data[:1000], vocab_file=None)
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
topic_df = tok.get_lda(n=100, num_words=5)
print("lda 이후 토픽테이블")
print(topic_df)
topic_df.to_json("lda_100.json")
print("LDA 완료")



## stock_vol 데이터 로드
stock_vol = pd.read_json("Data_crawler/dataset/삼성전자_trading_data/d_score.json")


## topic 기준 뉴스데이터 선정
ex = Ex(topic_df, stock_vol)

## stop 토픽 제거
print("스탑 토픽 정리")
ex.stop_topic(allow=3)
print("스탑토픽 이후 토픽 테이블")
print(ex.topic_df)

## 전 구간 내 토픽 카운팅
print("토픽 카운팅 실행")
ex.topic_count(news_df=tok.data[['date','text']], count_limit=20)
print("토픽 카운팅 결과 상위 50개 출력")
print(ex.topic_df.head(50))
print("토픽 카운트 데이터 저장")
ex.topic_df.to_json("topic_count.json")

## 일자별 테이블 생성
print("count_table 수행")
ex.topic_count_by_day(news_df=tok.data[['date','text']])
print("count_table 결과")
print(ex.count_table)
print("count_table 저장")
ex.count_table.to_json("count_table.json")

# print("이상없음")
## 피크데이 정의
ex.peak_day()

## 피크데이를 기준으로 FVE 추출
ex.get_FVE(cut_line=0.99)

## 선정 토픽과 거래량 상관관계 분석
pp = ex.corr_topic_vol(cut_line=0)
print(pp)
## 최종 토픽
pp = ex.topic_by_index(pp.index.tolist())
print("최종 사용토픽")
print(pp)

## 추출된 키워드를 키반 GPT 알고리즘을 통해 자연어 생성
