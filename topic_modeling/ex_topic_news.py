import pandas as pd
import numpy as np
from tqdm import tqdm


class Ex:

    STOP_LIST = {'코스피', '지수', '종목','강보합세','거래일','공모','공시','국채','나스닥','다우','매수','매도','배당','배당주','보합세','삼성동','성장주','액면','억','조','옵션','외국인','우리투자증권','우선주','유가증권','주가','주식','주주','증권가','증권거래소','증시','증자','천억','최고가','콜','풋','펀드','현물', '호가','휴장'}

    def __init__(self, topic_df, stock_vol):
        self.topic_df = topic_df
        self.stock_vol = stock_vol
        self.peak_day_news = None

    def div_news(self, month=6):

        pass

    def peak_day_vol(self, col_name="vol", top_perc=0.05, months=6):
        df_final = pd.DataFrame()
        first_date = self.stock_vol.index[0]
        end_date = self.stock_vol.index[-1]
        date_check = -1
        while end_date != date_check:
            last_date = first_date + pd.DateOffset(months=months)
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
        print(self.topic_df)
        self.topic_df = self.topic_df.drop(columns=['topic']).dropna().reset_index()

    # 전체 기사중 토픽관련 뉴스기사 갯수 카운트 by topic 기준
    def topic_count(self, news_df, topic_n=None, start_date=None, end_date=None):

        if start_date is not None:
            news_df = news_df[news_df['date'] >= start_date]
        if end_date is not None:
            news_df = news_df[news_df['date'] < end_date]
        if topic_n is not None:
            self.topic_df = self.topic_df[:topic_n]

        tqdm.pandas()
        self.topic_df['count'] = self.topic_df['allow_topic'].progress_apply(lambda x: news_df['text'].str.contains("^" + "".join(["(?=.*{})".format(i) for i in x]), regex=True).sum())
        self.topic_df = self.topic_df.sort_values(by=["count"],ascending=False).reset_index(drop=True)

        self.topic_df['index'] = self.topic_df.apply(lambda x: x.index+1).rename(columns={"index":"topic_imp"})

    # 일자별로 각 토픽에 관련된 뉴스 기사 갯수 카운트 토픽이 columns
    def topic_news_count(self, news, all_topic):



        # 토픽 3개 다 들어가있는 기사에 1 라벨링하기
        cnt_lis = []
        for a, b, c in tqdm(self.topic_df['allow_topic']):
            tmp = []
            for text in news_df['text']:
                if a in text and b in text and c in text:
                    tmp.append(1)
                else:
                    tmp.append(0)
            cnt_lis.append(tmp)

        # 기사 별로 라벨링된 리스트 news_df에 붙이기
        topic_count = pd.DataFrame(np.array(cnt_lis).T, columns=['topic' + str(n + 1) for n in range(len(self.topic_df))])
        total = pd.concat([news_df, topic_count], axis=1)

        # 날짜별로 그룹묶어서 count
        cnt_by_date = []
        for i in range(3):
            cnt_by_date.append(total['topic{}'.format(i + 1)].groupby(total['date']).sum())

        df = pd.DataFrame(cnt_by_date)
        count_df = df.transpose()

        return count_df