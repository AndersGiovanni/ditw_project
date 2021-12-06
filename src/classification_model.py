from typing import Dict, List, Union
from numpy.lib.function_base import average
from scipy.sparse.construct import random
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
import numpy as np
from src.json_utils import read_jsonl
from src.config import DATA_DIR
from src.annotation_agreement import encode_labels, split_annotated_data
from src.annotation_agreement import encode_labels, split_annotated_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm


def read_danish_stopwords() -> List[str]:
    with open('data/danish_stopwords.txt') as stop_file:
        return [word.strip() for word in stop_file.readlines()]


def tf_idf_vectotize(text: List[str], stopwords: List[str]) -> np.ndarray:
    tweet_tokenizer = TweetTokenizer().tokenize
    vectorizer = TfidfVectorizer(
        tokenizer=tweet_tokenizer,
        stop_words=stopwords,
        ngram_range=(
            1,
            3),
        min_df=3,
        max_df=0.80)
    return vectorizer.fit_transform(text)


def get_text_and_labels(annotations: List[Dict]) -> Union[List[str], List[str]]:
    texts = list()
    labels = list()

    for annotation in annotations:
        texts.append(annotation['data'])
        labels.append(annotation['label'][0])

    return texts, np.array(encode_labels(labels))


if __name__ == '__main__':

    annotation_file = DATA_DIR / 'annotations/post/annotated_data_v3.jsonl'
    annotations = read_jsonl(annotation_file)

    stopwords = read_danish_stopwords()

    evaluate_annotation_data, sentiment_classification_data = split_annotated_data(annotations)

    texts, y = get_text_and_labels(sentiment_classification_data)

    X = tf_idf_vectotize(texts, stopwords)

    #kf = KFold(n_splits=5)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_preds, all_trues = list(), list()
    idx = 1
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #clf = RandomForestClassifier(max_depth=2, random_state=42, n_jobs=-1)
        clf = LogisticRegression(random_state=42)

        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)

        f1 = f1_score(y_test, preds, average='weighted')

        print(f'Fold: {idx}, F1: {f1}')

        all_preds += preds.tolist()
        all_trues += y_test.tolist()

        idx += 1

    target_names = ['Negative', 'No Sentiment', 'Positive']
    print()
    print(classification_report(all_trues, all_preds, target_names=target_names))
