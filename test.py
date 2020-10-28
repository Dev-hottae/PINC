

# 1. 6개월 단위로 뉴스기사 분리


# 2. 각 기간별로 top 100 토픽 추출
# def lda_by_duration(

# 3. 불필요한 topic 리스트에 담고 제외

# 4. 일정 기간 나오지 못한 토픽은 우선제외 & 금융 관련 토픽 제외

# 5. 거래량 과거 100일 중 최처값 차감



###########################################
import os

import pandas as pd

from topic_modeling.ex_topic_news import Ex

topic_df = pd.read_json(r"C:\Users\dlagh\PycharmProjects\pp\PINC\lda_100.json", encoding='utf-8').reset_index(drop=True)
print("데이터 로딩 중")
data_dir = 'Data_crawler/dataset/삼성전자_pred'
data_list = os.listdir(data_dir)
news_df = pd.DataFrame([])
for data_file in data_list:
    news = pd.read_json(data_dir+"/"+data_file, encoding="utf-8")[['date','text']]
    news_df = news_df.append(news)
print("데이터 로딩 완료")
print(topic_df.head())
print(news_df.head())

stock_vol = 1
ex = Ex(topic_df, stock_vol)
ex.stop_topic(allow=3)
print("stop_topic after")
print(ex.topic_df)
ex.topic_count(news_df)
print("topic count after")
print(ex.topic_df)

# pp = ['애플', '에어컨', '오디오', '살균', '제거', '제품']
# STOP_LIST = {'코스피', '지수', '종목', '에어컨', '오디오'}
# print(type('에어컨'))
# print(type('에어컨'))
# print(set(pp).difference(STOP_LIST))
