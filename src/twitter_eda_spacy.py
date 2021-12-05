import spacy
from src.config import DATA_DIR
from src.json_utils import read_jsonl
import re
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.corpora as corpora
import gensim
from pprint import pprint

"""
Resources:
https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0

"""

with open('data/danish_stopwords.txt') as stop_file:
    stopwords = [word.strip() for word in stop_file.readlines()]


def remove_newlines(text: str) -> str:
    return re.sub(r'\n+', ' ', text)


def normalize_spaces(text: str) -> str:
    return re.sub(r'\s{2,}', ' ', text)


def remove_punctuation(text: str) -> str:
    return re.sub('[:,\\.!?\'\"\\”\\(\\)\\/\\-]', '', text)


def is_website_token(token: str) -> str:
    if token[:4] == 'http':
        return ''
    else:
        return token


preprocessing_func = [remove_punctuation, str.lower, is_website_token]

tweet_tokenizer = TweetTokenizer()

data = read_jsonl(DATA_DIR / 'dkpol_tweets.jsonl')
text = [i['text'] for i in data[:10]]
tokenized_text = [[token for token in tweet_tokenizer.tokenize(i) if token not in stopwords] for i in text]

all_tokenizer_text = []

for text in tokenized_text:
    post_precessed = []
    for token in text:
        for function in preprocessing_func:
            token = function(token)
        if token != '':
            post_precessed.append(token)
    all_tokenizer_text.append(post_precessed)

vectorizer = TfidfVectorizer(tokenizer=tweet_tokenizer, stop_words=None, ngram_range=(1, 3), min_df=5, max_df=0.90)
X = vectorizer.fit_transform(text)
