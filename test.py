#
#
# # 1. 6개월 단위로 뉴스기사 분리
#
#
# # 2. 각 기간별로 top 100 토픽 추출
# # def lda_by_duration(
#
# # 3. 불필요한 topic 리스트에 담고 제외
#
# # 4. 일정 기간 나오지 못한 토픽은 우선제외 & 금융 관련 토픽 제외
#
# # 5. 거래량 과거 100일 중 최처값 차감
#
#
# ##########################################
# from topic_modeling.ex_topic_news import Ex
# import pandas as pd
#
# def peak_day(stock_vol, months = 6):
#     stock_vol['date'] = stock_vol.date.astype('datetime64[ns]')
#     df_final = pd.DataFrame()
#     first_date = stock_vol.date[0]
#     end_date = stock_vol.iloc[-1]['date']
#     date_check = -1
#     while end_date != date_check:
#         last_date = first_date + pd.DateOffset(months = months)
#         df = stock_vol[(stock_vol.date >= first_date) & (stock_vol.date < last_date)]
#         date_check = df.iloc[-1]['date']
#         df = df.sort_values(by=['d_score'])
#         df = df[:int(len(df) * 0.95)].sort_values(by=['date'])
#         df_final = pd.concat([df_final, df])
#         first_date = last_date
#     return df_final.reset_index(drop=True)
#
#
# stock_vol = pd.read_json(r"C:\Users\dlagh\PycharmProjects\pp\PINC\Data_crawler\dataset\삼성전자_trading_data\d_score.json")
# stock_vol = stock_vol[stock_vol['date'] >= "2005-01-01"].reset_index(drop=True)
# count_table = pd.read_json(r"C:\Users\dlagh\PycharmProjects\pp\PINC\count_table.json", encoding='utf-8')
# ex = Ex(topic_df="", stock_vol=stock_vol)
# ex.count_table = count_table
# print("ex stock_vol")
# print(ex.stock_vol)
# print("ex count table")
# print(ex.count_table)
#
# print("ex peak day")
# ex.peak_day()
# print(ex.peak_vol)
#
# print("get fve")
# ex.get_FVE()
# print(ex.fve_topic_index)
#
# print("get corr")
# pp = ex.corr_topic_vol()
# print(pp)

import pandas as pd

from topic_modeling.ex_topic_news import Ex


count_table = pd.read_json(r"C:\Users\dlagh\PycharmProjects\pp\PINC\count_table.json")
stock_vol = pd.read_json(r"C:\Users\dlagh\PycharmProjects\pp\PINC\Data_crawler\dataset\삼성전자_trading_data\d_score.json")
topic_df = pd.read_json(r"C:\Users\dlagh\PycharmProjects\pp\PINC\topic_count.json", encoding='utf-8')
topic_df = topic_df.rename(columns={"index":"Topic"})
ex = Ex(topic_df="", stock_vol=stock_vol)
ex.count_table = count_table
ex.topic_df = topic_df
print(topic_df)
print("ex peak day 실행")
ex.peak_day()
print(ex.peak_vol)
print("----------")

print("get fve 실행")
ex.get_FVE()
print(ex.fve_topic_index)
print("-----------")

print("corr 실행")
pp = ex.corr_topic_vol(cut_line=0)
print(pp)

print("최종 추출토픽")
pp = ex.topic_by_index(pp.index.tolist())
print(pp)