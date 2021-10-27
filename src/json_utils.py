import json
from typing import Dict


def read_json(path: str) -> Dict:
    print(f'Reading {path}')
    with open(path, 'r') as infile:
        data: Dict = json.load(infile)
    return data


def write_json(container: Dict, filename: str) -> None:
    print(f'Writing to {filename}')
    with open(filename, 'w+') as outfile:
        json.dump(container, outfile)
