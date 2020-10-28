import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from tqdm import tqdm


class Ex:

    STOP_LIST = {'코스피', '지수', '종목','강보합세','거래일','공모','공시','국채','나스닥','다우','매수','매도','배당','배당주','보합세','삼성동','성장주','액면','억','조','옵션','외국인','우리투자증권','우선주','유가증권','주가','주식','주주','증권가','증권거래소','증시','증자','천억','최고가','콜','풋','펀드','현물', '호가','휴장'}

    def __init__(self, topic_df, stock_vol):
        self.topic_df = topic_df
        self.stock_vol = stock_vol
        self.count_table = None
        self.peak_day_news = None

    def div_news(self, month=6):

        pass

    def peak_day_vol(self, col_name="end", top_perc=0.05, months=6):
        self.stock_vol['date'] = pd.to_datetime(self.stock_vol['date'], format="%Y-%m-%d")
        self.stock_vol = self.stock_vol.set_index("date")
        df_final = pd.DataFrame()
        first_date = self.stock_vol.index[0]
        end_date = self.stock_vol.index[-1]
        date_check = -1
        print(self.stock_vol)
        while end_date != date_check:
            last_date = first_date + pd.DateOffset(months=months)
            print(first_date)
            print(last_date)
            df = self.stock_vol[first_date:last_date]
            date_check = df.index[-1]
            df = df.sort_values(by=[col_name])
            df = df[:int(len(df) * top_perc)].sort_index()
            df_final = pd.concat([df_final, df])
            first_date = last_date + pd.DateOffset(days=1)
        return df_final

    def peak_day_news(self):
        pass


    def stop_topic(self, allow=3):

        def _stop_check_func(topic_list, allow):
            allow_list = list(set(topic_list).difference(set(Ex.STOP_LIST)))
            if len(allow_list) >= allow:
                return allow_list
            else:
                return

        tqdm.pandas()
        self.topic_df['allow_topic'] = self.topic_df['topic'].progress_apply(lambda x: _stop_check_func(x, allow))
        self.topic_df = self.topic_df.drop(columns=['topic']).dropna().reset_index()
        self.topic_df['index'] = self.topic_df.index + 1

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
            return day_table[['index','count']].transpose().reset_index(drop=True)

        tqdm.pandas()
        self.topic_df['count'] = self.topic_df['allow_topic'].progress_apply(lambda x: news_df['text'].str.contains("^" + "".join(["(?=.*{})".format(i) for i in x]), regex=True).sum())
        self.topic_df = self.topic_df[self.topic_df['count'] >= count_limit].sort_values(by='count').reset_index(drop=True)
        self.topic_df['index'] = self.topic_df.index + 1

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

        self.count_table = count_table.reset_index()

    # count table 만들고 체크!!!!
    def get_FVE(self, topic, peak, n):

        peak_topic = pd.merge(topic, peak, on='date', how='inner')
        peak_topic = peak_topic.set_index('date')
        del peak_topic['volume']

        # 토픽별로 합계 테이블 작성
        df = pd.DataFrame(peak_topic.sum(axis=0), columns=['count'])

        # fve 칼럼 생성 후 내림차순 정렬
        df['fve'] = ''
        for i in range(len(df)):
            df['fve'][i] = df['count'][i] / sum(df['count'])
        df.sort_values(by='fve', ascending=False, inplace=True)

        # 누적합 칼럼 생성
        df['cumsum'] = df.fve.cumsum()

        # 누적합 n미만인 칼럼 추출
        final = df['cumsum'] < n
        new_df = df[final]

        return new_df