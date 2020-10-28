from tqdm import tqdm


class Ex:

    STOP_LIST = {'코스피', '지수', '종목', '에어컨', '오디오','이마트','증설','대통령'}

    def __init__(self, topic_df, stock_vol):
        self.topic_df = topic_df
        self.stock_vol = stock_vol

    def div_news(self, month=6):

        pass

    def get_lda(self):
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

    def topic_count(self, news_df, topic_n=None, start_date=None, end_date=None):

        if start_date is not None:
            news_df = news_df[news_df['date'] >= start_date]
        if end_date is not None:
            news_df = news_df[news_df['date'] < end_date]
        if topic_n is not None:
            self.topic_df = self.topic_df[:topic_n]

        tqdm.pandas()
        self.topic_df['count'] = self.topic_df['allow_topic'].progress_apply(lambda x: news_df['text'].str.contains("^" + "".join(["(?=.*{})".format(i) for i in x]), regex=True).sum())


