import torch
import torch.nn as nn
import pandas as pd
import faiss
from pathlib import Path
from .models.HYPERPARAMETERS import FREEZE_EMBEDDINGS, PROJECTION_DIM, MARGIN
from .models.core import TwoTowerModel
from .utils.load_data import load_word2vec
from .utils.preprocess_str import str_to_tokens


class SearchEngine:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self._initialize_resources()

    def _initialize_resources(self):
        # Load word2vec resources
        self.vocab, embeddings, self.word_to_idx = load_word2vec()

        # Initialize embedding layer
        self.embedding_layer = nn.Embedding.from_pretrained(
            embeddings, freeze=FREEZE_EMBEDDINGS
        )

        # Set dimensions
        self.embedding_dim = embeddings.shape[1]
        self.vocab_size = len(self.vocab)

        # Load dataset and index with correct paths
        data_path = self.base_dir / "data" / "training-with-tokens.parquet"
        index_path = self.base_dir / "data" / "doc-index-64.faiss"
        model_path = self.base_dir / "models" / "two_tower_state_dict.pth"

        self.df = pd.read_parquet(str(data_path))
        self.index = faiss.read_index(str(index_path))

        # Initialize model
        self.model = TwoTowerModel(
            embedding_dim=self.embedding_dim,
            projection_dim=PROJECTION_DIM,
            embedding_layer=self.embedding_layer,
            margin=MARGIN,
        )

        # Load model weights
        self.model.load_state_dict(
            torch.load(
                str(model_path),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )
        self.model.eval()  # Set model to evaluation mode

    def _get_nearest_neighbors(self, query: str, k: int = 5):
        query_tokens = torch.tensor([str_to_tokens(query, self.word_to_idx)])
        query_mask = (query_tokens != 0).float()

        with torch.no_grad():
            query_encoding = self.model.query_encode(query_tokens, query_mask)
            query_projection = self.model.query_project(query_encoding)

        query_vector = query_projection.detach().numpy()
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)

        documents = self.df.loc[indices.squeeze()]["doc_relevant"]
        urls = self.df.loc[indices.squeeze()]["url_relevant"]

        return documents, urls, distances

    def search(self, query: str):
        documents, urls, distances = self._get_nearest_neighbors(query)
        return documents.to_list(), distances.tolist()
