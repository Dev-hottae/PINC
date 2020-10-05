import re

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import requests
from dateutil.relativedelta import relativedelta
from matplotlib import font_manager as fm, rc
import matplotlib as mpl
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup


class DataMan():

    def __init__(self):
        self.stocks_info_path = r'/Mirae_data_research/data_set/stocks_info_raw.csv'
        self.stocks_price_history_path = r'C:\Users\dlagh\PycharmProjects\PINC\Mirae_data_research\data_set\stocks_price_history.csv'
        self.trade_train_path = r'C:\Users\dlagh\PycharmProjects\PINC\Mirae_data_research\data_set\trade_train.csv'


    # 필요 함수 세팅
    # 그룹과 일자 input 시 매수많이 일어난 top 추출함수
    def top_bid_stock(self, group, date, top):
        '''
        :param group: 그룹번호 ex) "01"
        :param date: ex) 201907
        :param top: 몇개 top? ex) 3
        :return:
        '''
        trade_history = pd.read_csv(self.trade_train_path, encoding='utf-8')
        data = trade_history[(trade_history['기준년월'] == date) & (trade_history['그룹번호'] == ("MAD" + group))]
        data = data.sort_values(by=["매수고객수"], ascending=False)
        data = data[data['매수고객수'] > 0][:top]

        stocks_info = pd.read_csv(self.stocks_info_path, encoding='utf-8').reset_index(drop=True)[['종목번호', '종목명', '20년7월TOP3대상여부', '시장구분','표준산업구분코드_대분류','표준산업구분코드_중분류','표준산업구분코드_소분류']]
        data = data.merge(stocks_info, how="inner", on="종목번호")
        return data

    # 종목명으로 종목코드 가져오기
    def code_by_name(self, name):
        '''
        :param name: 종목 이름
        :return:
        '''
        stocks_info = pd.read_csv(self.stocks_info_path, encoding='utf-8')
        return stocks_info.loc[stocks_info['종목명'] == name]['종목번호'].tolist()[0]

    # 종목코드로 종목명 가져오기
    def name_by_code(self, code):
        '''
        :param code: 종목 코드
        :return:
        '''
        stocks_info = pd.read_csv(self.stocks_info_path, encoding='utf-8')
        return stocks_info.loc[stocks_info['종목번호'] == code]['종목명'].tolist()[0]

    def visualizing_group_top(self, group, start_date, end_date, top=10):
        '''
        :param group: 그룹번호 ex) "01"
        :param start_date: ex) 201907
        :param end_date: ex) 202001
        :return:
        '''
        #################### Visual Studio Code 전용 한글 폰트 적용 ###################
        font_path = "C:/Windows/Fonts/H2GTRM.TTF"
        font_name = fm.FontProperties(fname=font_path).get_name()
        mpl.rc('font', family=font_name)
        ##############################################################################
        ######################### Colab 전용 한글 폰트 적용 ###########################
        # # 그래프에 retina display 적용
        # %config InlineBackend.figure_format = 'retina'
        # !apt -qq -y install fonts-nanum > /dev/null
        # fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
        # font = fm.FontProperties(fname=fontpath, size=10)
        # mpl.font_manager._rebuild()
        # # Colab 의 한글 폰트 설정
        # plt.rc('font', family='NanumBarunGothic')
        ##############################################################################

        trade_history = pd.read_csv(self.trade_train_path, encoding='utf-8')
        stocks_info = pd.read_csv(self.stocks_info_path, encoding='utf-8')

        data = trade_history[(trade_history['기준년월'] >= start_date) & (trade_history['기준년월'] < end_date) & (
                    trade_history['그룹번호'] == ("MAD" + group))]

        data = data.groupby(by=['종목번호'], as_index=False).sum()[["종목번호", "매수고객수"]]
        data = data.sort_values(by=["매수고객수"], ascending=False).reset_index(drop=True).head(top)
        data = data.merge(stocks_info[["종목번호", "종목명"]], how="inner", on="종목번호")
        plt.bar((data["종목명"]), data["매수고객수"])
        plt.xticks(rotation=40)
        plt.title("MAD{0} 종목명 별 매수고객수".format(group))
        plt.xlabel('종목명')
        plt.ylabel('매수고객수')
        plt.show()

    # 종목관련 데이터 비쥬얼라이징
    def visualizing_by_code(self, df):

        #################### Visual Studio Code 전용 한글 폰트 적용 ###################
        font_path = "C:/Windows/Fonts/H2GTRM.TTF"
        font_name = fm.FontProperties(fname=font_path).get_name()
        mpl.rc('font', family=font_name)
        ##############################################################################


    # 종목 변동성 구하기
    def get_std(self, code, start_month, end_month):

        print(code)

        stocks_price_history = pd.read_csv(self.stocks_price_history_path, encoding='utf-8')
        stocks_price_history['기준일자'] = pd.to_datetime(stocks_price_history['기준일자'], format="%Y%m%d")

        start = datetime.strptime(str(start_month), '%Y%m')
        end = datetime.strptime(str(end_month), '%Y%m')

        # 지정종목, 기간 데이터 추출
        data = stocks_price_history[(stocks_price_history['종목번호'] == code) & (stocks_price_history['기준일자'] >= start) & (stocks_price_history['기준일자'] < end)]

        # 거래량 0 제거
        data = data[data['거래량'] > 0]
        year = 252
        avg = data['종목종가'].mean()
        std_day = (data['종목종가'].std()/avg)/np.sqrt(len(data))
        stdrate_year = std_day * np.sqrt(year) * 100
        print(data['종목명'].tolist()[0], stdrate_year)

        return stdrate_year

    # 종목 모멘텀 구하기
    def get_momentom(self, code, start_month, end_month):

        stocks_price_history = pd.read_csv(self.stocks_price_history_path, encoding='utf-8')
        stocks_price_history['기준일자'] = pd.to_datetime(stocks_price_history['기준일자'], format="%Y%m%d")

        start = datetime.strptime(str(start_month), '%Y%m')
        end = datetime.strptime(str(end_month), '%Y%m')

        # 지정종목, 기간 데이터 추출
        data = stocks_price_history[(stocks_price_history['종목번호'] == code) & (stocks_price_history['기준일자'] >= start) & (
                    stocks_price_history['기준일자'] < end)]

        # 거래량 0 제거
        data = data[data['거래량'] > 0]

        close_prices = data['종목종가'].tolist()
        # print(close_prices)
        mom = np.log(close_prices[-1]/close_prices[0])*100
        print(data['종목명'].tolist()[0], mom)
        return mom


    # 가격변화률 구하기
    def get_change_rate(self, code, start_month, end_month):
        stocks_price_history = pd.read_csv(self.stocks_price_history_path, encoding='utf-8')
        stocks_price_history['기준일자'] = pd.to_datetime(stocks_price_history['기준일자'], format="%Y%m%d")

        start = datetime.strptime(str(start_month), '%Y%m')
        end = datetime.strptime(str(end_month), '%Y%m')

        # 지정종목, 기간 데이터 추출
        data = stocks_price_history[(stocks_price_history['종목번호'] == code) & (stocks_price_history['기준일자'] >= start) & (
                    stocks_price_history['기준일자'] < end)]

        start_price = data['종목종가'].tolist()[0]
        lowest_price = data['종목종가'].min()
        highest_price = data['종목종가'].max()

        return np.log(lowest_price/start_price)*100, np.log(highest_price/start_price)*100

    # 뉴스 출현빈도(1개월)
    def news_issue(self, stock_name, start_date):

        date_start = datetime.strptime(str(start_date), '%Y%m')
        date_end = date_start + relativedelta(months=+1) - relativedelta(day=+1)
        dot_date_start = date_start.strftime('%Y.%m.%d')
        dot_date_end = date_end.strftime('%Y.%m.%d')
        normal_date_start = date_start.strftime("%Y%m%d")
        normal_date_end = date_end.strftime("%Y%m%d")

        url = 'https://search.naver.com/search.naver?where=news&query={0}&sm=tab_opt&sort=0&photo=0&field=0&reporter_article=&pd=3&ds={1}&de={2}&docid=&nso=so%3Ar%2Cp%3Afrom{3}to{4}%2Ca%3Aall&mynews=0&refresh_start=0&related=0'.format(stock_name, dot_date_start, dot_date_end, normal_date_start, normal_date_end)

        res = requests.get(url)
        bs = BeautifulSoup(res.text, 'html.parser')
        data = bs.select("div.title_desc span")[0].text

        count = re.search("(?<=\/[\s])[\0-9]*", data).group(0)


        return count




data = DataMan()
# pp = data.get_change_rate("A297890", 202005, 202008)
# print(pp)

pp = data.news_issue("삼성전자", 202007)
print(pp)