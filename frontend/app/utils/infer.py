
import torch
import faiss

from .models.core import str_to_tokens
from .utils.load_data import load_word2vec

_, word_to_idx = load_word2vec() # Use embedding matrix word2vec-gensim-text8-custom-preprocess.model if you can

"""
query: A string
model: A TwoTowerEncoder model based on HYPERPARAMETERS.yaml
df: A dataframe with documents and urls, from training-with-tokens.parquet
index: A faiss index, loaded from doc-index-64.faiss
"""

# Function to get nearest neighbors
def get_nearest_docs(query, model, df, index, k=5):
    query_tokens = torch.tensor([str_to_tokens(query, word_to_idx)])
    query_mask = (query_tokens != 0).float()
    query_encoding = model.query_encode(query_tokens, query_mask)
    query_projection = model.query_project(query_encoding)

    query_vector = query_projection.detach().numpy()
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, k)

    documents = df.loc[indices.squeeze()]['doc_relevant']
    urls = df.loc[indices.squeeze()]['url_relevant']

    return documents, urls, distances