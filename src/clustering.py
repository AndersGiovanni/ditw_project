from transformers.utils.dummy_pt_objects import AutoModel
from src.json_utils import read_json, read_jsonl
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModel
import torch

data = read_jsonl('data/dkpol_tweets.jsonl')

#texts = [tweet['text'] for tweet in data]

tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
model = AutoModel.from_pretrained("Maltehb/danish-bert-botxo")

inputs = tokenizer(data, return_tensors="pt", padding=True)
outputs = model(**inputs, output_hidden_states=True)

hidden = outputs.hidden_states[-1]

embeddings = list()

for emb in hidden:
    embeddings.append(torch.tensor(emb[0, :]))

embeddings = torch.stack(embeddings)

umap_embeddings = umap.UMAP(n_neighbors=15,
                            n_components=5,
                            metric='cosine').fit_transform(embeddings)

cluster = hdbscan.HDBSCAN(min_cluster_size=5,
                          metric='euclidean',
                          cluster_selection_method='eom').fit(umap_embeddings)

# Prepare data
umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
result = pd.DataFrame(umap_data, columns=['x', 'y'])
result['labels'] = cluster.labels_

# Visualize clusters
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.75)
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=1.0, cmap='hsv_r')
plt.colorbar()
plt.show()
