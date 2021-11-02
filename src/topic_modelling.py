
from sentence_transformers import SentenceTransformer
import umap
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
from sklearn.cluster import DBSCAN
import numpy as np

from src.json_utils import read_json, read_jsonl

data = read_jsonl('data/dkpol_tweets.jsonl')
model = SentenceTransformer('Maltehb/-l-ctra-danish-electra-small-cased')
embeddings = model.encode(data, show_progress_bar=True, normalize_embeddings=True)

umap_embeddings = umap.UMAP().fit_transform(embeddings)

cluster = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                          gen_min_span_tree=True, leaf_size=40,
                          metric='euclidean', min_cluster_size=5, min_samples=None, p=None).fit(embeddings)

# Prepare data
umap_data = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
result = pd.DataFrame(umap_data, columns=['x', 'y'])
result['labels'] = cluster.labels_

# Visualize clusters
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.5)
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.5, cmap='hsv_r')
plt.colorbar()
plt.show()
