

# 1. 6개월 단위로 뉴스기사 분리


# 2. 각 기간별로 top 100 토픽 추출
# def lda_by_duration(

# 3. 불필요한 topic 리스트에 담고 제외

# 4. 일정 기간 나오지 못한 토픽은 우선제외 & 금융 관련 토픽 제외

# 5. 거래량 과거 100일 중 최처값 차감


##########################################
from topic_modeling.ex_topic_news import Ex
import pandas as pd

topic = pd.read_json(r"C:\Users\dlagh\PycharmProjects\pp\PINC\topic_count.json", encoding='utf-8')
vol = pd.read_csv(r"C:\Users\dlagh\PycharmProjects\pp\PINC\Data_crawler\dataset\삼성전자_trading_data\d_score.csv")

ex = Ex(topic, vol)
print(ex.peak_day_vol(col_name='d_score'))

