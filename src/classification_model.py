import pickle
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
from danlp.models import load_bert_tone_model
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.annotation_agreement import encode_labels, split_annotated_data
from src.config import DATA_DIR
from src.json_utils import read_jsonl


def read_danish_stopwords() -> List[str]:
    with open('data/danish_stopwords.txt') as stop_file:
        return [word.strip() for word in stop_file.readlines()]


def tf_idf_vectotize(train: List[str], test: List[str], stopwords: List[str],
                     save_vectorizer_name: str = '') -> Union[np.ndarray, np.ndarray]:
    tweet_tokenizer = TweetTokenizer().tokenize
    vectorizer = TfidfVectorizer(
        tokenizer=tweet_tokenizer,
        stop_words=stopwords,
        ngram_range=(
            1,
            3),
        min_df=3,
        max_df=0.80)
    train_vec = vectorizer.fit_transform(train)
    test_vec = vectorizer.transform(test)

    if save_vectorizer_name:
        with open(save_vectorizer_name, 'wb') as fout:
            pickle.dump(vectorizer, fout)

    return train_vec, test_vec


def get_text_and_labels(annotations: List[Dict]) -> Union[List[str], List[str]]:
    texts = list()
    labels = list()

    for annotation in annotations:
        texts.append(annotation['data'])
        labels.append(annotation['label'][0])

    return texts, np.array(encode_labels(labels))


def evaluate_BERT_tone(texts: List[str], labels: List[str]) -> None:

    classifier = load_bert_tone_model()

    target_names = ['Positive', 'No Sentiment', 'Negative']

    predictions = []

    for text in tqdm(texts):
        largest_proba_id = np.argmax(classifier.predict_proba(text)[0])
        predictions.append(largest_proba_id)

    print(classification_report(labels, predictions, target_names=target_names))


def train_and_save_model(X: np.ndarray, y: np.ndarray, modelname: str = 'sentiment_log_reg.sav'):
    model = LogisticRegression()
    model.fit(X, y)
    # save the model to disk
    pickle.dump(model, open(modelname, 'wb'))
    print(f'Model saved to: {modelname}')


if __name__ == '__main__':

    annotation_file = DATA_DIR / 'annotations/post/annotated_data_v3.jsonl'
    annotations = read_jsonl(annotation_file)

    stopwords = read_danish_stopwords()

    evaluate_annotation_data, sentiment_classification_data = split_annotated_data(annotations)

    texts, y = get_text_and_labels(sentiment_classification_data)
    X, _ = tf_idf_vectotize(texts, texts, stopwords, save_vectorizer_name='models/tf_idf_vectorizer.pkl')

    train_and_save_model(X, y, 'models/sentiment_log_reg.pkl')

    exit()

    # kf = KFold(n_splits=5)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds, all_trues = list(), list()
    idx = 1

    wrong_preds = []

    for train_index, test_index in kf.split(texts, y):

        X_train_text, X_test_text = np.array(texts)[train_index], np.array(texts)[test_index]
        X_train, X_test = tf_idf_vectotize(X_train_text, X_test_text, stopwords)
        y_train, y_test = y[train_index], y[test_index]

        # clf = RandomForestClassifier(max_depth=2, random_state=42, n_jobs=-1)
        clf = LogisticRegression(random_state=42)

        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)

        f1 = f1_score(y_test, preds, average='weighted')

        wrong_preds_split = [(X_test_text[idx], pred, y_test[idx]) for idx, pred in enumerate(preds) if pred != y_test[idx]]
        wrong_preds.extend((wrong_preds_split))

        print(f'Fold: {idx}, F1: {f1}')

        all_preds += preds.tolist()
        all_trues += y_test.tolist()

        idx += 1

    with open('data/wrong_preds.txt', 'w') as wrongs:
        for line in wrong_preds:
            wrongs.write(f'{list(*line)}\n')

    exit()

    target_names = ['Negative', 'Neutral', 'Positive']
    print()
    print(classification_report(all_trues, all_preds, target_names=target_names))

    cm = confusion_matrix(all_trues, all_preds, labels=clf.classes_, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=target_names)
    disp.plot(cmap=plt.get_cmap('Blues'))
    # plt.tight_layout()
    # plt.savefig('data/img/cm_v1.png', dpi=300)
    plt.show()

    print()
    # print('BERT Tone Sentiment Classification')
    # evaluate_BERT_tone(texts, y)
