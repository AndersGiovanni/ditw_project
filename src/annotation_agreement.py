from typing import Dict, List, Union
from statsmodels.stats.inter_rater import fleiss_kappa
from src.config import DATA_DIR
from src.json_utils import read_jsonl
from collections import Counter
import numpy as np


def encode_labels(labels: List[str]) -> List[str]:

    labels_mapping = {
        'NEGATIV': 0,
        'NO SENTIMENT': 1,
        'POSITIV': 2
    }

    new_labels = []
    for label in labels:
        new_labels.append(labels_mapping[label])

    return new_labels


def count_annotations(annotations: Dict[str, List[str]]) -> None:

    names = list(annotations.keys())

    for name in names:
        print(f'{name}: {Counter(annotations[name])}')


def compute_fleiss_kappa(annotations: Dict[str, List[str]]) -> None:

    names = list(annotations.keys())

    list_of_annotations = []

    for a1, a2, a3, a4 in zip(annotations[names[0]], annotations[names[1]], annotations[names[2]], annotations[names[3]]):
        list_of_annotations.append(encode_labels([a1, a2, a3, a4]))

    # The input of fleiss is the count of each label pr row i.e. [3, 1, 0]
    # means 3 negative, 1 no sentiment and 0 postive
    fleiss_kappa_list_format = [[0, 0, 0] for i in range(len(list_of_annotations))]

    for row, tweet_annotations in enumerate(list_of_annotations):
        for column, count in list(Counter(tweet_annotations).items()):
            fleiss_kappa_list_format[row][column] = count

    print(fleiss_kappa(fleiss_kappa_list_format))


def split_annotated_data(annotated_data: List[Dict]) -> Union[List[Dict], List[Dict]]:

    ids_for_fleiss_kappa = [i[0] for i in list(Counter([i['id'] for i in annotated_data]).most_common(50)) if i[1] == 4]

    evaluate_annotation_data = list()

    sentiment_classification_data = list()

    for annotation in annotated_data:
        id_ = annotation['id']

        if id_ in ids_for_fleiss_kappa:
            evaluate_annotation_data.append(annotation)
        else:
            sentiment_classification_data.append(annotation)

    return evaluate_annotation_data, sentiment_classification_data


def preprocess_annotations(annotations: List[Dict]) -> Dict[str, List[str]]:

    collect_annotations = {
        'name1': [],
        'name2': [],
        'name3': [],
        'name4': []
    }

    for annotation in annotations:
        label = annotation['label'][0]
        annotator = annotation['annotation']
        collect_annotations[annotator].append(label)

    return collect_annotations


if __name__ == '__main__':

    annotation_file = DATA_DIR / 'annotations/post/annotated_data_v2.jsonl'
    annotations = read_jsonl(annotation_file)

    evaluate_annotation_data, sentiment_classification_data = split_annotated_data(annotations)

    collect_annotations = preprocess_annotations(evaluate_annotation_data)

    count_annotations(collect_annotations)

    compute_fleiss_kappa(collect_annotations)
