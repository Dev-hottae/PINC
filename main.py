
### 전체 프로세스

## 00시 00분 scrapy crawler 작동


## 크롤링 종료 후 데이터 전처리
from Data_crawler.data_pre_processing import Pre

path = r"C:\Users\dlagh\Desktop\final_project"
pre = Pre(keyword="삼성전자", path=path)
pre.run_processing()

## TOPIC 모델링을 통한 1일 핵심 주제 키워드 추출


## 추출된 키워드를 키반 GPT 알고리즘을 통해 자연어 생성