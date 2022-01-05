from pathlib import Path
from typing import List, Optional, Union
import re
import torch
import numpy as np
import csv
import os


def replace_url(texts: List[str], replacement: Optional[str] = 'URL') -> List[str]:
    '''Replace URLs in a tweet by a token. The URLs in tweets has the format: https://t.co/afQGDec3Es'''

    regex = re.compile(r'(https:\/\/t.co\/\w+)', re.IGNORECASE)
    texts_without_url = []
    for line in texts:
        line = regex.sub(replacement, line)
        texts_without_url.append(line)

    return texts_without_url


def write_tsv_files(
    folder_path: Path,
    vectors: Union[torch.tensor, np.ndarray],
    metadata: Union[List[str], List[List[str]]],
    metadata_labels: List[str],
) -> None:
    """Writes two .tsv files, a file of vectors and a file with metadata.
    These files are used in Tensorflow Projector.
    """

    assert (
        os.path.isdir(folder_path) is True
    ), f"{folder_path} doesn't exist. Please make sure that the folder exists"

    embeddings_path: Path = folder_path / "embeddings.tsv"
    metadata_path: Path = folder_path / "metadata.tsv"

    print(f"Writing embeddings to {embeddings_path} and metadata to {metadata_path}")

    # Write embeddings
    with open(embeddings_path, "w") as fw:
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.cpu().detach().numpy()
        csv_writer = csv.writer(fw, delimiter="\t")
        csv_writer.writerows(vectors)

    # Write metadata
    with open(metadata_path, "w") as file:
        labels_count: int = len(metadata_labels)
        metadata_labels = "\t".join(metadata_labels)

        if labels_count > 1:
            file.write(f"{metadata_labels}\n")
            metadata = ["\t".join(list(i)) for i in metadata]

        for meta in metadata:
            file.write(f"{meta}\n")
