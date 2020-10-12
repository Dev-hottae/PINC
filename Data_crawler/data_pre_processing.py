import pandas as pd
import numpy as np
import re
import os

class Pre():

    BASIC_PATH = ""

    def __init__(self, keyword, path='Data_crawler/dataset', min_str=400, max_str=2500):

        Pre.BASIC_PATH = path

        # 키워드
        self.keyword = keyword

        # 기사 글자 수 제한
        self.min_str = min_str
        self.max_str = max_str

        # 뉴스데이터 저장경로
        self.mk_path = ''
        self.mn_path = ''
        self.yn_path = ''
        self.yi_path = ''
        self.id_path = ''
        self.fn_path = ''
        self.hk_path = ''
        self.hr_path = ''
        self.ak_path = ''

        # 전처리된 데이터 담을 경로
        # 디렉토리 생성
        try:
            os.mkdir(Pre.BASIC_PATH + "/{}_pred".format(keyword))
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

        self.rpath_pred = Pre.BASIC_PATH + "/{}_pred".format(keyword)
        # raw 데이터 담긴 경로
        self.rdata_path = Pre.BASIC_PATH + "/{}_raw".format(keyword)

        # 전처리 진행
        self.data_merge_div()

    # 합치고 중복제거 & 뉴스사별로 파일분리
    def data_merge_div(self):

        # 뉴스사별로 분리
        ## 우선 통합
        data_list = os.listdir(self.rdata_path)

        news_data = pd.DataFrame([])
        for data_path in data_list:
            print(self.rdata_path + "/" + data_path)
            data = pd.read_json(self.rdata_path + "/" + data_path, encoding='utf-8')
            news_data = news_data.append(data)

        # 기사공백제거
        news_data = news_data[news_data['text'] != ""]
        # 기사중복제거
        news_data = news_data.drop_duplicates(subset=['url'])
        news_data = news_data.drop_duplicates(subset=['text'])

        # 뉴스사별로 분리
        mk = news_data[news_data['office'] == "매일경제"]
        fn = news_data[news_data['office'] == "파이낸셜뉴스"]
        yi = news_data[news_data['office'] == "연합인포맥스"]
        yn = news_data[news_data['office'] == "연합뉴스"]
        id = news_data[news_data['office'] == "이데일리"]
        mn = news_data[news_data['office'] == "머니투데이"]
        hr = news_data[news_data['office'] == "헤럴드경제"]
        hk = news_data[news_data['office'] == "한국경제"]
        ak = news_data[news_data['office'] == "아시아경제"]

        # 저장
        mk.to_csv(self.rpath_pred + "/" + "{}_매일경제.csv".format(self.keyword), encoding="utf-8")
        fn.to_csv(self.rpath_pred + "/" + "{}_파이낸셜뉴스.csv".format(self.keyword), encoding="utf-8")
        yi.to_csv(self.rpath_pred + "/" + "{}_연합인포맥스.csv".format(self.keyword), encoding="utf-8")
        yn.to_csv(self.rpath_pred + "/" + "{}_연합뉴스.csv".format(self.keyword), encoding="utf-8")
        id.to_csv(self.rpath_pred + "/" + "{}_이데일리.csv".format(self.keyword), encoding="utf-8")
        mn.to_csv(self.rpath_pred + "/" + "{}_머니투데이.csv".format(self.keyword), encoding="utf-8")
        hr.to_csv(self.rpath_pred + "/" + "{}_헤럴드경제.csv".format(self.keyword), encoding="utf-8")
        hk.to_csv(self.rpath_pred + "/" + "{}_한국경제.csv".format(self.keyword), encoding="utf-8")
        ak.to_csv(self.rpath_pred + "/" + "{}_아시아경제.csv".format(self.keyword), encoding="utf-8")

        # 기존 raw 데이터 삭제
        ###

        # 각 뉴스사 경로
        self.mk_path = self.rpath_pred + "/" + "{}_매일경제.csv".format(self.keyword)
        self.mn_path = self.rpath_pred + "/" + "{}_머니투데이.csv".format(self.keyword)
        self.yn_path = self.rpath_pred + "/" + "{}_연합뉴스.csv".format(self.keyword)
        self.yi_path = self.rpath_pred + "/" + "{}_연합인포맥스.csv".format(self.keyword)
        self.id_path = self.rpath_pred + "/" + "{}_이데일리.csv".format(self.keyword)
        self.fn_path = self.rpath_pred + "/" + "{}_파이낸셜뉴스.csv".format(self.keyword)
        self.hk_path = self.rpath_pred + "/" + "{}_한국경제.csv".format(self.keyword)
        self.hr_path = self.rpath_pred + "/" + "{}_헤럴드경제.csv".format(self.keyword)
        self.ak_path = self.rpath_pred + "/" + "{}_아시아경제.csv".format(self.keyword)

        print("뉴스사별 csv 파일 생성완료")

    # 전처리
    def run_processing(self):
        self._ak_pre()
        print("아시아경제 전처리완료")
        self._hr_pre()
        print("헤럴드경제 전처리완료")
        self._yn_pre()
        print("연합뉴스 전처리완료")
        self._yi_pre()
        print("연합인포맥스 전처리완료")
        self._id_pre()
        print("이데일리 전처리완료")
        self._fn_pre()
        print("파이낸스뉴스 전처리완료")
        self._mk_pre()
        print("매일경제 전처리완료")

        # ## 퀄리티가 너무 떨어지는 관계로 일단 보류
        # self._mn_pre()
        # print("머니투데이 전처리완료")
        # self._hk_pre()
        # print("한국경제 전처리완료")

        # 분할파일 삭제
        ###

    # 아시아경제
    def _ak_pre(self):
        ak_df = pd.read_csv(self.ak_path).sort_values(by=['date'])

        # [아시아경제 조호윤 기자] <cut>
        ak_df['text'] = ak_df['text'].str.replace('\[[\w\W]*(아시아경제)[\w\W]+\]', '').replace('\n', ' ').replace('\\',
                                                                                                              ' ').replace(
            '\"', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')

        # 괄호데이터 삭제
        ak_df['text'] = ak_df['text'].str.replace(r'\([^\)]*\)', '')

        # 글자 수 col 추가가
        ak_df['count_str'] = ak_df['text'].str.count("[\s\S]", re.I)

        # 적정글자수 기사 추출
        ak_df = ak_df[ak_df['count_str'] >= self.min_str]
        ak_df = ak_df[ak_df['count_str'] <= self.max_str]

        # 파일저장
        ak_df.to_json(self.rpath_pred + "/" + 'pre_{0}_아시아경제.json'.format(self.keyword))

    # 헤럴드경제
    def _hr_pre(self):
        hr_df = pd.read_csv(self.hr_path).sort_values(by=['date'])

        # 이메일 형식 제거
        hr_df['text'] = hr_df['text'].str.replace('[A-z0-9]+@heraldcorp.com', '')

        # [00해럴드경제00] <cut>
        hr_df['text'] = hr_df['text'].str.replace('\[[\w\W]*(헤럴드경제)[\w\W]+\]', '')
        hr_df['text'] = hr_df['text'].str.replace('\[[\w\W]*(헤럴드경제)[\w\W]*］', '')
        hr_df['text'] = hr_df['text'].str.replace('\[[\w\W]*=[\w\W]+기자\]', '')

        # 괄호데이터 삭제
        hr_df['text'] = hr_df['text'].str.replace(r'\([^\)]*\)', '')

        # 맨 뒷줄 제거
        hr_df['match'] = hr_df['text'].str.match(pat="[\w\W]+(?=\.[^.]*$)")
        hr_df = hr_df[hr_df['match'] == True].drop(columns=['match'])
        hr_df['text'] = hr_df['text'].str.extract("([\w\W]+(?=\.[^.]*$))").replace('\n', ' ').replace('\\', ' ').replace('\"', ' ').replace( '\r', ' ').replace('\t', ' ').replace('  ', ' ')

        # 글자 수 추가
        hr_df['count_str'] = hr_df['text'].str.count("[\s\S]", re.I)

        # 적정글자수 기사 추출
        hr_df = hr_df[hr_df['count_str'] >= self.min_str]
        hr_df = hr_df[hr_df['count_str'] <= self.max_str]

        # 파일저장
        hr_df.to_json(self.rpath_pred + "/" + 'pre_{0}_헤럴드경제.json'.format(self.keyword))

    # 연합뉴스
    def _yn_pre(self):

        yn_df = pd.read_csv(self.yn_path).sort_values(by=['date'])

        # 기사형식 가진 기사들만 추출
        yn_df = yn_df[yn_df['text'].str.contains("\([가-힣]+=연합[가-힣]+\)[가-힣\s]+=", case=False)]

        # ~~~ (OO=연합ㅇㅇ) ㅇㅇㅇ 기자 = <cut>
        yn_df['text'] = yn_df['text'].str.replace('^[\w\W]*\([가-힣]+=연합[가-힣]+\)[가-힣\s]+=', '')
        # 이메일이후 <cut>
        yn_df['text'] = yn_df['text'].str.replace('([A-Za-z0-9]+@(yna.))(?!.*([A-Za-z0-9]+@(yna.)).*)[\w\W]+',
                                                  '').replace('\n', ' ').replace('\\', ' ').replace('\"', ' ').replace(
            '\r', ' ').replace('\t', ' ').replace('  ', ' ')
        # 저작권자(c) 연합뉴스 이후 <cut>
        yn_df['text'] = yn_df['text'].str.replace('(저작권자\(c\)[\w\W]+)', '').replace('\n', ' ').replace('\\',
                                                                                                       ' ').replace(
            '\"', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')

        # 괄호데이터 삭제
        yn_df['text'] = yn_df['text'].str.replace(r'\([^\)]*\)', '')
        yn_df['text'] = yn_df['text'].str.replace(r'\[[0-9]+\]', '')

        # 글자 수 추가가
        yn_df['count_str'] = yn_df['text'].str.count("[\s\S]", re.I)

        yn_df = yn_df[yn_df['count_str'] >= self.min_str]
        yn_df = yn_df[yn_df['count_str'] <= self.max_str]

        # 파일저장
        yn_df.to_json(self.rpath_pred + "/" + 'pre_{0}_연합뉴스.json'.format(self.keyword))

    # 연합인포맥스
    def _yi_pre(self):
        yi_df = pd.read_csv(self.yi_path).sort_values(by=['date'])

        # 기사형식 가진 기사들만 추출
        yi_df = yi_df[yi_df['text'].str.contains("\([가-힣]+=연합[가-힣]+\)[가-힣\s]+=", case=False)]

        # (OO=연합ㅇㅇ) ㅇㅇㅇ 기자 = <cut>
        yi_df['text'] = yi_df['text'].str.replace('^[\w\W]*\([가-힣]+=연합[가-힣]+\)[가-힣\s]+=', '')
        # 이메일이후 <cut>
        yi_df['text'] = yi_df['text'].str.replace('([A-Za-z0-9]+@(yna.))(?!.*([A-Za-z0-9]+@(yna.)).*)[\w\W]+',
                                                  '').replace('\n', ' ').replace('\\', ' ').replace('\"', ' ').replace(
            '\r', ' ').replace('\t', ' ').replace('  ', ' ')
        # 저작권자(c) 연합뉴스 이후 <cut>
        yi_df['text'] = yi_df['text'].str.replace('(저작권자[\s]*©[\s]*연합인포맥스)[\w\W]*', '').replace('\n', ' ').replace('\\',
                                                                                                                   ' ').replace(
            '\"', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')
        # 괄호데이터 삭제
        yi_df['text'] = yi_df['text'].str.replace(r'\([^\)]*\)', '')
        yi_df['text'] = yi_df['text'].str.replace(r'\[[0-9]+\]', '')

        # 글자 수 추가
        yi_df['count_str'] = yi_df['text'].str.count("[\s\S]", re.I)

        yi_df = yi_df[yi_df['count_str'] >= self.min_str]
        yi_df = yi_df[yi_df['count_str'] <= self.max_str]

        # 파일저장
        yi_df.to_json(self.rpath_pred + "/" + 'pre_{0}_연합인포맥스.json'.format(self.keyword))

    # 이데일리
    def _id_pre(self):

        id_df = pd.read_csv(self.id_path).sort_values(by=['date'])

        # 자바스크립트제거
        id_df['text'] = id_df['text'].str.replace(r'[\s]+\(function[\w\W]+true}\);[\s]+', ' ')

        # 기사형식 가진 기사들만 추출
        id_df = id_df[id_df['text'].str.contains("^\[[가-힣\=]*이데일리[가-힣\s]+기자[^\]]*\]", case=False)]

        # ~[이데일리 김윤지 기자] <cut>
        id_df['text'] = id_df['text'].str.replace('^\[[가-힣\=]*이데일리[가-힣\s]+기자[^\]]*\]', '').replace('\n', ' ').replace(
            '\\', ' ').replace('\"', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')

        # \xa0 삭제
        id_df['text'] = id_df['text'].str.replace(r'\xa0', ' ')

        # 괄호데이터 삭제
        id_df['text'] = id_df['text'].str.replace(r'\([^\)]*\)', '')
        id_df['text'] = id_df['text'].str.replace(r'\[[0-9]+\]', '')

        # 글자수 기준 기사 제거
        id_df['count_str'] = id_df['text'].str.count("[\s\S]", re.I)

        # 적정글자수 기사 추출
        id_df = id_df[id_df['count_str'] >= self.min_str]
        id_df = id_df[id_df['count_str'] <= self.max_str]

        # 파일저장
        id_df.to_json(self.rpath_pred + "/" + 'pre_{0}_이데일리.json'.format(self.keyword))

    # 파이낸셜뉴스
    def _fn_pre(self):

        fn_df = pd.read_csv(self.fn_path).sort_values(by=['date'])

        # 마지막줄 이메일 삭제
        fn_df['text'] = fn_df['text'].str.replace(r'[/]*[A-z0-9]+@fnnews.com[\w\W]*', '')
        # 괄호데이터 삭제
        fn_df['text'] = fn_df['text'].str.replace(r'\([^\)]*\)', '')
        fn_df['text'] = fn_df['text'].str.replace(r'^\[파이낸셜뉴스\]', '')
        fn_df['text'] = fn_df['text'].str.replace(r'^【[\w\W]+】 ', '')

        # 글자 수 추가
        fn_df['count_str'] = fn_df['text'].str.count("[\s\S]", re.I)

        # 적정글자수 기사 추출
        fn_df = fn_df[fn_df['count_str'] >= self.min_str]
        fn_df = fn_df[fn_df['count_str'] <= self.max_str]

        # 파일저장
        fn_df.to_json(self.rpath_pred + "/" + 'pre_{0}_파이낸셜뉴스.json'.format(self.keyword))

    # 매일경제
    def _mk_pre(self):

        mk_df = pd.read_csv(self.mk_path).sort_values(by=['date'])

        # 매경좋은기사 패턴
        mk_df = mk_df[mk_df['text'].str.contains("\[[/=가-힣\s]+\][\s]\[ⓒ 매일경제 & mk.co.kr, 무단전재 및 재배포 금지\]$", case=False)]

        # 마지막 [ㅇㅇㅇ기자][무단 재배포 금지 footer 제거]
        mk_df['text'] = mk_df['text'].str.replace(r'\[[/=가-힣\s]+\][\s]\[ⓒ 매일경제 & mk.co.kr, 무단전재 및 재배포 금지\]$',
                                                  '').replace('\n', ' ').replace('\\', ' ').replace('\"', ' ').replace(
            '\r', ' ').replace('\t', ' ').replace('  ', ' ')
        # 괄호데이터 삭제
        mk_df['text'] = mk_df['text'].str.replace(r'\([^\)]*\)', '')
        # 매경 TEST 기사 제외
        mk_df = mk_df[mk_df['title'].str.contains(r'매경TEST', case=False) == False]
        mk_df = mk_df[mk_df['title'].str.contains(r'\[인사\]', case=False) == False]

        # 글자 수 추가
        mk_df['count_str'] = mk_df['text'].str.count("[\s\S]", re.I)

        # 적정글자수 기사 추출
        mk_df = mk_df[mk_df['count_str'] >= self.min_str]
        mk_df = mk_df[mk_df['count_str'] <= self.max_str]

        # 파일저장
        mk_df.to_json(self.rpath_pred + "/" + 'pre_{0}_매일경제.json'.format(self.keyword))

    # 머니투데이
    def _mn_pre(self):
        pass

    # 한국경제
    def _hk_pre(self):
        pass