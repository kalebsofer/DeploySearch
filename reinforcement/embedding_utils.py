import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import numpy as np
from pathlib import Path
import io

# Download required NLTK data
nltk.download("stopwords")
nltk.download("punkt")

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


# Preprocessing functions
def preprocess_list(tokens: list[str]) -> list[str]:
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    tokens = ["[S]"] + tokens + ["[E]"]
    return tokens


def preprocess_query(query: str) -> list[str]:
    if query is None:
        return []

    query = query.lower()
    tokens = simple_preprocess(query, deacc=True)
    tokens = preprocess_list(tokens)
    return tokens


def str_to_tokens(s: str, word_to_idx: dict[str, int]) -> list[int]:
    split = preprocess_query(s)
    return [word_to_idx[word] for word in split if word in word_to_idx]


# Model classes
class DocumentProjection(nn.Module):
    def __init__(self, input_size, embedding_dim):
        super(DocumentProjection, self).__init__()
        self.fc1 = nn.Linear(input_size, embedding_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, doc_encoding):
        x = self.fc1(doc_encoding)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class QueryProjection(nn.Module):
    def __init__(self, encoding_dim, projection_dim):
        super(QueryProjection, self).__init__()
        self.fc1 = nn.Linear(encoding_dim, encoding_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(encoding_dim, projection_dim)

    def forward(self, doc_encoding):
        x = self.fc1(doc_encoding)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class TwoTowerModel(nn.Module):
    def __init__(self, embedding_dim, embedding_layer, projection_dim, margin):
        super().__init__()

        self.embedding = embedding_layer
        self.encoding_dim = embedding_dim
        self.doc_projection = DocumentProjection(embedding_dim, projection_dim)
        self.query_projection = QueryProjection(embedding_dim, projection_dim)
        self.margin = margin

    def doc_encode(self, doc_ids, doc_mask=None):
        doc_embed = self.embedding(doc_ids)
        doc_embed = (
            doc_embed * doc_mask.unsqueeze(-1) if doc_mask is not None else doc_embed
        )
        doc_encoding = doc_embed.mean(dim=1)
        return doc_encoding

    def query_encode(self, query_ids, query_mask=None):
        query_embed = self.embedding(query_ids)
        query_embed = (
            query_embed * query_mask.unsqueeze(-1)
            if query_mask is not None
            else query_embed
        )
        query_encoding = query_embed.mean(dim=1)
        return query_encoding

    def doc_project(self, doc_encoding):
        return self.doc_projection(doc_encoding)

    def query_project(self, query_encoding):
        return self.query_projection(query_encoding)

    def compare_projections(self, d_projection, q_projection):
        return F.cosine_similarity(d_projection, q_projection, dim=1)


class EmbeddingProcessor:
    def __init__(self, minio_client):
        self.minio_client = minio_client
        self._initialize_resources()

    def _get_file_from_minio(self, bucket: str, object_name: str) -> bytes:
        try:
            response = self.minio_client.get_object(bucket, object_name)
            return response.read()
        except Exception as e:
            raise Exception(f"Error loading {object_name} from MinIO: {str(e)}")

    def load_word2vec(self):
        """Load word2vec model from MinIO storage."""
        model_name = "word-vector-embeddings.model"
        try:
            # Load from MinIO
            model_bytes = self._get_file_from_minio("data", model_name)

            # Save temporarily and load with gensim
            temp_path = Path("/tmp") / model_name
            with open(temp_path, "wb") as f:
                f.write(model_bytes)

            w2v = Word2Vec.load(str(temp_path))
            temp_path.unlink()  # Clean up

            # Extract embeddings
            vocab = w2v.wv.index_to_key
            word_to_idx = {word: i for i, word in enumerate(vocab)}
            embeddings_array = np.array([w2v.wv[word] for word in vocab])
            embeddings = torch.tensor(embeddings_array, dtype=torch.float32)

            return vocab, embeddings, word_to_idx

        except Exception as e:
            raise Exception(f"Failed to load word2vec model: {str(e)}")
        finally:
            if "temp_path" in locals() and temp_path.exists():
                temp_path.unlink()

    def _initialize_resources(self):
        # Load word vectors
        self.vocab, embeddings, self.word_to_idx = self.load_word2vec()

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True)

        self.embedding_dim = embeddings.shape[1]
        self.vocab_size = len(self.vocab)

        # Initialize model
        self.model = TwoTowerModel(
            embedding_dim=self.embedding_dim,
            embedding_layer=self.embedding_layer,
            projection_dim=64,
            margin=0.5,
        )

        # Load latest model weights
        model_bytes = self._get_file_from_minio("data", "two_tower_state_dict.pth")
        model_buffer = io.BytesIO(model_bytes)
        self.model.load_state_dict(
            torch.load(
                model_buffer,
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )
        self.model.eval()

    def get_embeddings(self, query: str, document: str):
        """Get embeddings for a query-document pair"""
        # Convert to tokens
        query_tokens = torch.tensor([str_to_tokens(query, self.word_to_idx)])
        doc_tokens = torch.tensor([str_to_tokens(document, self.word_to_idx)])

        # Create masks
        query_mask = (query_tokens != 0).float()
        doc_mask = (doc_tokens != 0).float()

        with torch.no_grad():
            # Get encodings
            query_encoding = self.model.query_encode(query_tokens, query_mask)
            doc_encoding = self.model.doc_encode(doc_tokens, doc_mask)

            # Get projections
            query_projection = self.model.query_project(query_encoding)
            doc_projection = self.model.doc_project(doc_encoding)

        return query_projection, doc_projection
