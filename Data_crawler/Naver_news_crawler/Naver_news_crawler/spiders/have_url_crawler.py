import re

import scrapy
import pandas as pd


class HaveUrl(scrapy.Spider):
    name = "have_url_crawler"

    def __init__(self, file_path='', file_type='', news_office='', time_break=0, **kwargs):
        super().__init__(**kwargs)
        self.file_path = file_path
        self.time_break = time_break
        
        # self.news_office = news_office
        self.news_office ='아시아경제'
        # # url 설정
        # if file_type == 'json':
        #     self.start_urls = pd.read_json(self.file_path)['url'].tolist()
        # elif file_type == 'csv':
        #     self.start_urls = pd.read_csv(self.file_path)['url'].tolist()

        self.start_urls=['https://view.asiae.co.kr/article/2020092511181299253']

    def parse(self, response):

        news_url = response.request.url

        if self.news_office == '연합뉴스':
            yield scrapy.Request(news_url, callback=self.article_yn)
        elif self.news_office == '연합인포맥스':
            yield scrapy.Request(news_url, callback=self.article_yi)
        elif self.news_office == '이데일리':
            yield scrapy.Request(news_url, callback=self.article_ed)
        elif self.news_office == '머니투데이':
            yield scrapy.Request(news_url, callback=self.article_mn)
        elif self.news_office == '아시아경제':
            yield scrapy.Request(news_url, callback=self.article_ak)
        elif self.news_office == '헤럴드경제':
            yield scrapy.Request(news_url, callback=self.article_hr)
        elif self.news_office == '파이낸셜뉴스':
            yield scrapy.Request(news_url, callback=self.article_fn)
        elif self.news_office == '한국경제':
            yield scrapy.Request(news_url, callback=self.article_hk)
        elif self.news_office == '매일경제':
            yield scrapy.Request(news_url, callback=self.article_mk)
        else:
            pass

    # 연합인포맥스 url에 ' ' 포함
    def article_yi(self, response):
        yield {
            'date': re.search('[0-9]{4}[\.\-]?[0-9]{2}[\.\-]?[0-9]{2}',
                              response.css('div.info-text ul.no-bullet li::text').getall()[1]).group(),
            'office': '연합인포맥스',
            'url': response.url,
            'text': re.sub('(\<[^\<\>]*\>)', ' ',
                           response.css('div#article-view-content-div').getall()[0]).strip().replace('\n',
                                                                                                     ' ').replace(
                '\\', ' ').replace('\"', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')
        }

    # 연합뉴스 url에 ' ' 포함
    def article_yn(self, response):
        yield {
            'date': re.search('[0-9]{4}[\.\-]?[0-9]{2}[\.\-]?[0-9]{2}',
                              response.css('p.update-time::text').get()).group(),
            'office': '연합뉴스',
            'url': response.url,
            'text': re.sub('(\<[^\<\>]*\>)', ' ', response.css('div.story-news').getall()[0]).strip().replace('\n',
                                                                                                              ' ').replace(
                '\\', ' ').replace('\"', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')
        }

    # 이데일리 url에 ' ' 포함
    def article_ed(self, response):

        try:
            first_text = re.search("[가-힣]+(?=\([0-9]+\))", response.css('div.news_body a.topmenu_textq::text').getall()[0]).group()
        except:
            first_text = ''

        text = response.css('div.news_body::text').getall()

        yield {
            'date': re.search('[0-9]{4}[\.\-]?[0-9]{2}[\.\-]?[0-9]{2}',
                              response.css('div.dates ul li p::text').getall()[0]).group(),
            'office': '이데일리',
            'url': response.url,
            'text': text[0].strip() + first_text + ' '.join(text[1:]).strip().replace('\n', ' ').replace('\\',
                                                                                                              ' ').replace(
                '\"', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')
        }

    # 머니투데이
    def article_mn(self, response):

        yield {
            'date': re.search('[0-9]{4}[\.\-]?[0-9]{2}[\.\-]?[0-9]{2}',
                              response.css('div#article div.info ul.info2 li.date::text').get()).group(),
            'office': '머니투데이',
            'url': response.url,
            'text': ' '.join(response.css('div#textBody *::text').getall()).strip().replace('\n', ' ').replace('\\',
                                                                                                             ' ').replace(
                '\"', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')
        }

    # 아시아경제
    def article_ak(self, response):

        print("!!!!!!!!!!!!!!!!!!")
        print(response.xpath('//*[@id="txt_area"]/p/text() | //*[@id="txt_area"]/p[1]/span/a/text()').getall())
        print("!!!!!!!!!!!!!!!!!!!")
        # ' '.join(t.strip() for i, t in enumerate(response.css('p ::text').extract()) if i < 2).strip()

        yield {
            'date': re.search('[0-9]{4}[\.\-]?[0-9]{2}[\.\-]?[0-9]{2}',
                              response.css('div.area_title p.user_data::text').getall()[1]).group(),
            'office': '아시아경제',
            'url': response.url,
            'text': ' '.join(response.xpath('//*[@id="txt_area"]/p/text() | //*[@id="txt_area"]/p[1]/span/a/text()').getall()).strip().replace('\n',
                                                                                              ' ').replace(
                '\\',
                ' ').replace(
                '\"', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')
        }

    # 헤럴드경제
    def article_hr(self, response):
        yield {
            'date': re.search('[0-9]{4}[\.\-]?[0-9]{2}[\.\-]?[0-9]{2}',
                              ' '.join(response.css('div.view_top_t2 ul li.ellipsis::text').getall())).group(),
            'office': '헤럴드경제',
            'url': response.url,
            'text': ' '.join(response.css('div#content_ADTOM div#articleText > p::text').getall()).strip().replace(
                '\n',
                ' ').replace(
                '\\',
                ' ').replace(
                '\"', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')
        }

    # 파이낸셜뉴스
    def article_fn(self, response):
        yield {
            'date': re.search('[0-9]{4}[\.\-]?[0-9]{2}[\.\-]?[0-9]{2}',
                              response.css('div.view_hd div.byline em::text').getall()[1]).group(),
            'office': '파이낸셜뉴스',
            'url': response.url,
            'text': ' '.join(response.css('div#article_content::text').getall()).strip().replace('\n',
                                                                                                 ' ').replace(
                '\\',
                ' ').replace(
                '\"', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')
        }

    # 한국경제
    def article_hk(self, response):

        print("!!!!!!!!!!!!")
        print(response.css('div#articletxt ::text').getall())
        print("!!!!!!!!!!")

        yield {
            'date': re.search('[0-9]{4}[\.\-]?[0-9]{2}[\.\-]?[0-9]{2}',
                              response.css('div#container div.date_info span.num::text').get()).group(),
            'office': '한국경제',
            'url': response.url,

            # ' '.join(t.strip() for i, t in enumerate(response.css('div#articletxt ::text').extract()) if i < 2).strip()

            'text': ' '.join(response.css('div#articletxt ::text').getall()).strip().replace('\n',
                                                                                            ' ').replace(
                '\\',
                ' ').replace(
                '\"', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')
        }

    # 매일경제
    def article_mk(self, response):
        yield {
            'date': re.search('[0-9]{4}[\.\-]?[0-9]{2}[\.\-]?[0-9]{2}',
                              response.css('li.lasttime::text').get()).group(),
            'office': '매일경제',
            'url': response.url,
            'text': ' '.join(response.css('div#article_body div.art_txt::text').getall()).strip().replace('\n',
                                                                                                          ' ').replace(
                '\\',
                ' ').replace(
                '\"', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')
        }