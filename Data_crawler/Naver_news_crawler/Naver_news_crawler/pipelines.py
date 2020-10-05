# -*- coding: utf-8 -*-
# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from __future__ import unicode_literals
from scrapy.exporters import JsonItemExporter, CsvItemExporter
import sys

# # JSON 파일로 저장하는 클래스
class NaverNewsCrawlerPipeline(object):
    def __init__(self):
        self.file = open("another_copied.json", 'wb')
        self.exporter = JsonItemExporter(self.file, encoding='utf-8')
        self.exporter.start_exporting()

    def close_spider(self, spider):
        self.exporter.finish_exporting()
        self.file.close()

    def process_item(self, item, spider):
        self.exporter.export_item(item)
        return item