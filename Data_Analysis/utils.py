import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from PINC.Data_Analysis.kogpt2.utils import download, tokenizer, get_tokenizer
from gluonnlp.data import SentencepieceTokenizer


def sentencePieceTokenizer():
    tok_path = get_tokenizer()
    sentencepieceTokenizer = SentencepieceTokenizer(tok_path)

    return sentencepieceTokenizer

def koGPT2Vocab():
    cachedir = '~/kogpt2/'
    # download vocab
    vocab_info = tokenizer
    vocab_path = download(vocab_info['url'],
                        vocab_info['fname'],
                        vocab_info['chksum'],
                        cachedir=cachedir)

    koGPT2_vocab = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                                mask_token=None,
                                                                sep_token=None,
                                                                cls_token=None,
                                                                unknown_token='<unk>',
                                                                padding_token='<pad>',
                                                                bos_token='<s>',
                                                                eos_token='</s>')
    return koGPT2_vocab

def Dataset_Train(path):
    data_path = path
    news_data = pd.read_json(data_path, encoding='utf-8')
    dataset_train = []
    for i in tqdm(range(len(news_data))):
        dataset_train.append(news_data['Topic_keywords'][i] + ". " + news_data['Text'][i])
    return dataset_train

class GPTDataset(Dataset):
    def __init__(self, data_file, vocab, tokenizer):
        self.data =[]
        self.vocab = vocab
        self.tokenizer = tokenizer

    for data in data_file:
        tokenized_line = self.tokenizer(data)
        if len(tokenized_line) <= 1020: # 문장 총길이 1022로 제한
            index_of_words = [vocab.bos_token] + tokenized_line + [vocab.padding_token] * (1020 - len(tokenized_line)) + [vocab.eos_token]
            self.data.append(vocab(index_of_words))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        item = self.data[index]
        return item
