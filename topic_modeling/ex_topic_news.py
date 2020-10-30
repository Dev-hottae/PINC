import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from tqdm import tqdm


class Ex:

    STOP_LIST = {'코스피', '지수', '종목','강보합세','거래일','공모','공시','국채','나스닥','다우','매수','매도','배당','배당주','보합세','삼성동','성장주','액면','억','조','옵션','외국인','우리투자증권','우선주','유가증권','주가','주식','주주','증권가','증권거래소','증시','증자','천억','최고가','콜','풋','펀드','현물', '호가','휴장'}

    def __init__(self, topic_df, stock_vol):
        self.topic_df = topic_df
        self.stock_vol = stock_vol
        self.peak_vol = None
        self.count_table = None
        self.fve_topic_index = None
        self.best_topic_index = None

    def peak_day(self, col_name="d_score", months=6, cut_line=0.05):
        df_final = pd.DataFrame()
        first_date = self.stock_vol.date[0]
        end_date = self.stock_vol.iloc[-1]['date']
        date_check = -1
        while end_date != date_check:
            last_date = first_date + pd.DateOffset(months=months)
            df = self.stock_vol[(self.stock_vol.date >= first_date) & (self.stock_vol.date < last_date)]
            date_check = df.iloc[-1]['date']
            df = df.sort_values(by=[col_name], ascending=False)
            df = df[:int(len(df) * cut_line)].sort_values(by=['date'])
            df_final = pd.concat([df_final, df])
            first_date = last_date

        self.peak_vol = df_final.reset_index(drop=True)

    def topic_by_index(self, index_list):
        results = list(map(int, index_list))
        print(results)
        print(self.topic_df)
        return self.topic_df[self.topic_df['t_index'].isin(results)][['t_index','allow_topic']]

    def stop_topic(self, allow=3):

        def _stop_check_func(topic_list, allow):
            allow_list = list(set(topic_list).difference(set(Ex.STOP_LIST)))
            if len(allow_list) >= allow:
                return allow_list
            else:
                return

        tqdm.pandas()
        self.topic_df['allow_topic'] = self.topic_df['topic'].progress_apply(lambda x: _stop_check_func(x, allow))
        self.topic_df = self.topic_df.drop(columns=['topic']).dropna().reset_index(drop=True)
        self.topic_df['t_index'] = self.topic_df.index + 1

    # 전체 기사중 토픽관련 뉴스기사 갯수 카운트 by topic 기준
    def topic_count(self, news_df, topic_n=None, start_date=None, end_date=None, count_limit=1, inner=False):

        if start_date is not None:
            news_df = news_df[news_df['date'] >= start_date]
        if end_date is not None:
            news_df = news_df[news_df['date'] < end_date]
        if topic_n is not None:
            self.topic_df = self.topic_df[:topic_n]

        if inner is True:
            day_table = self.topic_df.copy()
            day_table['count'] = self.topic_df['allow_topic'].apply(lambda x: news_df['text'].str.contains("^" + "".join(["(?=.*{})".format(i) for i in x]), regex=True).sum())
            return day_table[['t_index','count']].transpose().reset_index(drop=True)

        tqdm.pandas()
        self.topic_df['count'] = self.topic_df['allow_topic'].progress_apply(lambda x: news_df['text'].str.contains("^" + "".join(["(?=.*{})".format(i) for i in x]), regex=True).sum())
        self.topic_df = self.topic_df[self.topic_df['count'] >= count_limit].sort_values(by='count').reset_index(drop=True)


    # 일자별로 각 토픽에 관련된 뉴스 기사 갯수 카운트 토픽이 columns
    def topic_count_by_day(self, news_df):

        start_date = news_df['date'].min()
        end_date = news_df['date'].max()

        date_list = pd.date_range(start=start_date, end=end_date)

        count_table = pd.DataFrame([])
        print("토픽 카운트!!!")
        for date in tqdm(date_list):
            date_plus1 = date + relativedelta(days=+1)
            result = self.topic_count(news_df, start_date=date, end_date=date_plus1, count_limit=0, inner=True)
            result = result.rename(columns=result.iloc[0]).drop(result.index[0]).reset_index(drop=True)
            result.insert(0, 'date', date)

            ## T 된 column index, count 테이블이 result 에 담김
            count_table = count_table.append(result)

        self.count_table = count_table.reset_index(drop=True)

    # count table 만들고 체크!!!!
    def get_FVE(self, cut_line=0.9):
        peak_count = pd.merge(self.peak_vol[['date']], self.count_table, right_on='date', left_on='date')
        df = pd.DataFrame({'Topic': peak_count.columns[1:], 'FVE': round(peak_count.sum() / sum(peak_count.sum()), 5)})
        df = df.sort_values('FVE', ascending=False).reset_index(drop=True)
        df['Cumsum'] = df.FVE.cumsum()
        df = df[df.Cumsum <= cut_line]

        # 주요 토픽 인덱스만 저장
        self.fve_topic_index = df['Topic'].tolist()


    # 거래량과의 상관관계 계산
    def corr_topic_vol(self, check_col='d_score', cut_line=0.3):
        corr_df = self.stock_vol[['date', check_col]]
        corr_df = corr_df.merge(self.count_table[['date'] + self.fve_topic_index], how='inner', left_on='date',
                                right_on='date')
        corr_df = corr_df.corr()[[check_col]][1:].rename(columns={check_col: "corr"})
        corr_df = corr_df[corr_df['corr']>=cut_line]

        self.best_topic_index = corr_df.index.tolist()
        return corr_df