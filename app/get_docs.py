from .utils.load_data import load_word2vec
from .utils.preprocess_str import str_to_tokens
import torch.nn as nn
from .models.core import DocumentDataset, TwoTowerModel, loss_fn
import pandas as pd
import faiss
import torch
from .models.HYPERPARAMETERS import FREEZE_EMBEDDINGS, PROJECTION_DIM, MARGIN

# Function to get nearest neighbors
def get_nearest_neighbors(query, model, df, k=5, index='', word_to_idx=''):
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

def get_docs(q):
    # Load embeddings
    vocab, embeddings, word_to_idx = load_word2vec()
    embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=FREEZE_EMBEDDINGS)

    EMBEDDING_DIM = embeddings.shape[1]
    VOCAB_SIZE = len(vocab)

    df = pd.read_parquet('data/training-with-tokens.parquet')
    index = faiss.read_index('data/doc-index-64.faiss')
    model = TwoTowerModel(
        embedding_dim=EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM,
        embedding_layer=embedding_layer,
        margin=MARGIN
    )

    # Fixed loading with weights_only=True
    model.load_state_dict(
        torch.load(
            f'models/two_tower_state_dict.pth',
            weights_only=True,  # Add this parameter
            map_location=torch.device('cpu')  # Add this to ensure CPU loading
        )
    )

    documents, urls, distances = get_nearest_neighbors(q, model, df, 5, index, word_to_idx)
    doc=documents.to_list()
    url=urls.to_list()
    rel_docs_sim=distances.tolist()
    return doc, rel_docs_sim

if __name__ == "__main__":
    docs,urls,distances=get_docs("What is the capital of France?")
    print(docs)
