import asyncio
import datetime
import re
from datetime import time
from xml.etree.ElementTree import ParseError

import pandas as pd
import unicodedata
from hanspell import spell_checker
from tqdm import tqdm
from gensim import corpora
from gensim.models import LdaModel, TfidfModel
import warnings
from eunjeon import Mecab

warnings.filterwarnings("ignore")
import os
from gensim.models.wrappers import LdaMallet

class Tokenizer:

    def __init__(self, data, vocab_file):
        self.data = data
        self.vocab_file = vocab_file

        pass

    def date_condition(self, start_date="2005-01-01"):
        self.data = self.data[self.data['date'] >= start_date].sort_values(by=['date']).reset_index(drop=True)[['date', 'text']]


    # 네이버 맞춤법 체크 // 이후에 멀티스레드 등의 방식으로 진행해야함
    def check_spell(self, data_trunc=False, max_char=500, dropna=True):
        '''
        :param df_text: 맞춤법 교정할 dataframe
        :param max_char: 최대 교정가능 글자수 (500)
        :return:
        '''

        async def pre_func(text):
            # 띄어쓰기 전체 합침
            pattern = re.compile(r'\s+')
            sentence = re.sub(pattern, '', text).strip()

            # 맞춤법을 위해 500자 이전 . 까지 concat // . 좌측이 숫자가 아니면 okay
            for_ch = re.match(r".+[^0-9]\.", sentence[:max_char]).group()

            try:
                ch_text = spell_checker.check(for_ch).checked
            except ParseError:
                return None

            if for_ch == ch_text:
                return None

            # 네이버 500자 맞춤법 교정
            print("final : ", ch_text)
            return ch_text

        tqdm.pandas()
        now = datetime.datetime.now()
        if data_trunc > 0:
            self.data['text'] = self.data['text'][:data_trunc].progress_apply(lambda x: asyncio.run(pre_func(x)))
        else:
            self.data['text'] = self.data['text'].progress_apply(lambda x: asyncio.run(pre_func(x)))
        print(datetime.datetime.now()-now)
        # now = datetime.datetime.now()
        # if data_trunc > 0:
        #     self.data['text'] = self.data['text'][:data_trunc].progress_apply(lambda x: pre_func(x))
        # else:
        #     self.data['text'] = self.data['text'].progress_apply(lambda x: pre_func(x))
        # print(datetime.datetime.now()-now)

        if dropna is True:
            self.data = self.data.dropna().reset_index(drop=True)

    def ex_stopword(self, bool, allow_type="nv"):

        if allow_type == "n":
            allow_vv = ["NNG", "NNP", "NNB", "NR", "NP"]
        elif allow_type == "nv":
            allow_vv = ["NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "VX", "VCP", "VCN", "MM", "MAG", "MAJ"]

        tqdm.pandas()
        if bool is True:
            mecab = Mecab()
            self.data['stopped_text'] = self.data['text'].progress_apply(lambda x: " ".join([word[0] for word in mecab.pos(str(x)) if word[1] in allow_vv]))
        else:
            self.data['stopped_text'] = self.data['text']

    def train_vocab(self, tokenizer, vocab_size=32000, min_freq=5):

        self.data['stopped_text'].to_csv("topic_modeling/for_tokenize.txt", encoding='utf-8')

        # Initialize a tokenizer
        tokenizer = tokenizer

        # Then train it!
        tokenizer.train(
            files=["topic_modeling/for_tokenize.txt"],
            vocab_size=vocab_size,
            min_frequency=min_freq,
            show_progress=True,
            )

        # And finally save it somewhere
        tokenizer.save_model(".", "topic_modeling/huggingface_tokenizer_kor_32000")

    def tokenizer(self, tokenizer):
        tqdm.pandas()
        self.data['token'] = self.data['stopped_text'].progress_apply(lambda x: tokenizer.encode(x).tokens)

        # 토큰화 이후 ##제거
        self._token_refine()
        return self.data

    def _token_refine(self):

        self.data['token'] = self.data['token'].progress_apply(lambda x: " ".join(x))
        self.data['token'] = self.data['token'].str.replace(r'##[^\s]*[\s]', '')
        self.data['token'] = self.data['token'].str.replace(r'__[^\s]*[\s]', '')
        self.data['token'] = self.data['token'].str.replace(r'\[[\w]{3}\]', '')
        # 분절 자모 재결합
        self.data['token'] = self.data['token'].progress_apply(lambda x: unicodedata.normalize('NFC', x))

        self.data['token'] = self.data['token'].progress_apply(lambda x: x.split())


    def get_lda(self, n, num_words):
        id2word = corpora.Dictionary(self.data['token'])
        corpus_TDM = [id2word.doc2bow(doc) for doc in self.data['token']]
        tfidf = TfidfModel(corpus_TDM)
        corpus_TFIDF = tfidf[corpus_TDM]
        lda = LdaModel(corpus=corpus_TFIDF,
                            id2word=id2word,
                            num_topics=n,
                            random_state=100)

        twords = {}
        for topic, word in lda.print_topics(n, num_words=num_words):
            twords["topic_{}".format(topic+1)] = [re.findall(r"(?<=\")[^\s][^(?=\")]*(?=\")", word)]

        return pd.DataFrame(twords).T.rename(columns={0:"topic"})

    def mallet_lda(self, num):

        id2word = corpora.Dictionary(self.data['token'])
        texts = self.data['token']
        corpus = [id2word.doc2bow(text) for text in texts]
        os.environ['Mallet_HOME'] = 'C:\\Mallet'
        mallet_path = 'C:\\Mallet\\bin\\mallet'
        ldamallet = LdaMallet(mallet_path, corpus=corpus, num_topics=num, id2word=id2word)
        return ldamallet.print_topics(num, num_words=6)
