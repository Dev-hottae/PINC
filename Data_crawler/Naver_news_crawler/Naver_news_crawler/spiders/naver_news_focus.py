import time
import scrapy
import re
import datetime
import pandas as pd
import tqdm

class NewsFocusSpider(scrapy.Spider):

    name = "naver_news_focus"

    HOST = 'https://finance.naver.com'
    url_format = 'https://finance.naver.com/news/news_list.nhn?mode=LSS2D&section_id=101&section_id2=258&date={0}&page={1}'

    def __init__(self, start_date, end_date=None, time_break=0, **kwargs):

        super().__init__(**kwargs)

        if end_date is None:
            end_date = datetime.datetime.today().strftime("%Y%m%d")

        self.time_break = time_break
        self.start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
        self.end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
        self.cur_page = 1
        self.last_page = 0

        self.start_urls = [
            NewsFocusSpider.url_format.format(start_date, self.cur_page)
        ]
    def parse(self, response):

        # 현 날짜 마지막 페이지 번호
        try:
            self.last_page = re.search("(?<=page=)[0-9]+", response.css('table.Nnavi td td.pgRR a::attr(href)').get()).group()
        except:
            self.last_page = self.last_page

        # 이번페이지 크롤링
        articles_link = response.css('div#contentarea ul.realtimeNewsList dl > .articleSubject')
        articles_datas = response.css('div#contentarea ul.realtimeNewsList dl > .articleSummary')

        for link, datas in zip(articles_link, articles_datas):

            title = link.css('a::text').get()
            # 타이틀이 [표]로 시작하지 않으면
            if title[:3] != "[표]":

                # 타임 브레이커
                time.sleep(self.time_break)
                press = datas.css('span.press::text').get()

                date = re.search("[0-9]{4}[\.\-]?[0-9]{2}[\.\-]?[0-9]{2}", datas.css('span.wdate::text').get()).group()
                news_url = NewsFocusSpider.HOST + link.css('a::attr(href)').get()

                # 데이터 요청
                request = scrapy.Request(news_url, callback=self.news_call)
                request.meta['press'] = press
                request.meta['date'] = date
                request.meta['url'] = news_url
                request.meta['title'] = title

                yield request

            # 타이틀이 [표]로 시작하면
            else:
                pass

        # 마지막 페이지 도착 이전까지
        if self.cur_page < int(self.last_page):
            # 다음페이지 정보로 업데이트
            self.cur_page += 1
            # 다음페이지 호출
            str_date = self.start_date.strftime("%Y%m%d")
            next_page_url = 'https://finance.naver.com/news/news_list.nhn?mode=LSS2D&section_id=101&section_id2=258&date={0}&page={1}'.format(
                str_date, str(self.cur_page))

            yield scrapy.Request(next_page_url, callback=self.parse)

        else:
            # 페이지 없음
            # 날짜 업데이트
            self.start_date += datetime.timedelta(days=1)
            if self.start_date <= self.end_date:
                # 신규 날짜로 1번 페이지 부터 재호출
                self.cur_page = 1
                str_date = self.start_date.strftime("%Y%m%d")
                next_date_url = 'https://finance.naver.com/news/news_list.nhn?mode=LSS2D&section_id=101&section_id2=258&date={0}&page={1}'.format(
                    str_date, str(self.cur_page))

                yield scrapy.Request(next_date_url, callback=self.parse)

            else:
                print("크롤링 종료")
                return

    def news_call(self, response):

        yield {
            'date': response.meta['date'],
            'office': response.meta['press'],
            'title': response.meta['title'],
            'url' : response.meta['url'],

            'text': re.sub('(\<[^\<\>]*\>)|▶[\s\S]+', ' ',
                           response.css('div#content').getall()[0]).strip().replace('\n', ' ').replace(
                '\\', ' ').replace('\"', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')
        }
