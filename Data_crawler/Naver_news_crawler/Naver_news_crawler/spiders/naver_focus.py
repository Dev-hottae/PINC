import time
import scrapy
import re
import datetime
import pandas as pd
import tqdm

class NewsFocusSpider(scrapy.Spider):

    name = "naver_focus"

    HOST = 'https://finance.naver.com'
    
    url = 'https://finance.naver.com/news/news_list.nhn?mode=LSS3D&section_id=101&section_id2=258&section_id3=401&date={}&page={}'
    
    now_page = 1
    temp = ''
    start_date = datetime.datetime(2020, 8, 30)
    now_date = ''

    def start_requests(self):
        yield scrapy.Request(NewsFocusSpider.url.format(str(NewsFocusSpider.start_date.date()).replace('-',''), NewsFocusSpider.now_page), callback=self.parse)

    def parse(self, response):
        titles = response.css('ul.realtimeNewsList dl .articleSubject a::text').getall()
        sources = response.css('ul.realtimeNewsList dl .articleSubject a::attr(href)').getall()
        
        if NewsFocusSpider.now_page == 1:
            NewsFocusSpider.now_date = re.search('(?<=date\=)[0-9]+', response.url).group()
        # 페이지 정보가 없으면
        if titles == []:
            NewsFocusSpider.now_page = 1
            yield scrapy.Request(NewsFocusSpider.HOST + response.css('div.pagenavi_day a::attr(href)').getall()[2], callback=self.parse)

            return

        for title, source in zip(titles, sources):
            pass_this = False
            # 크롤링 제외 타이틀
            for nu in ['[표]', '[마감]', '[출발]']:
                if nu in title:
                    pass_this = True
                    break
                    
            if not pass_this:
                request = scrapy.Request(NewsFocusSpider.HOST + source, callback=self.crawl_news)
                request.meta['title'] = title
                request.meta['url'] = response.url
                yield request

        NewsFocusSpider.now_page += 1
        yield scrapy.Request(NewsFocusSpider.url.format(NewsFocusSpider.now_date, NewsFocusSpider.now_page), callback=self.parse)


    def crawl_news(self, response):

        yield {
            'date': response.css('span.article_date::text').get(),
            'title' : response.meta['title'],
            'office': response.css('span.press img::attr(title)').get(),
            'url' : response.meta['url'],
            'text': re.sub('(\<[^\<\>]*\>)|▶[\s\S]+', ' ',
                    response.css('div#content').getall()[0]).strip().replace('\n', ' ').replace('\\', ' ').replace('\"', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')
        }
