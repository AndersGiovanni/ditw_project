import json
from typing import Dict, List


def read_json(path: str) -> Dict:
    print(f'Reading {path}')
    with open(path, 'r') as infile:
        data: Dict = json.load(infile)
    return data


def write_json(container: Dict, filename: str) -> None:
    print(f'Writing to {filename}')
    with open(filename, 'w+') as outfile:
        json.dump(container, outfile)


def read_jsonl(path: str) -> List[Dict]:
    print(f'Reading {path}')
    data = []
    with open(path, 'r') as infile:
        for line in infile.readlines()[:1000]:
            line = json.loads(line)
            data.append(line['text'].strip())
    return data
