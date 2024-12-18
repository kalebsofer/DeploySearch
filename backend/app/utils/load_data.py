import torch
import gensim
import numpy as np
import pandas as pd
from pathlib import Path
from minio import Minio
from minio.error import S3Error
from ..config.settings import get_settings

settings = get_settings()


class ModelLoadError(Exception):
    """Raised when the word2vec model cannot be loaded from MinIO."""

    pass


def get_minio_client():
    return Minio(
        settings.MINIO_URL,
        access_key=settings.MINIO_ROOT_USER,
        secret_key=settings.MINIO_ROOT_PASSWORD,
        secure=settings.MINIO_SECURE,
    )


def load_word2vec(random_seed=42):
    """
    Load word2vec model from MinIO storage.

    Args:
        random_seed (int): Seed for reproducibility

    Returns:
        tuple: (vocab, embeddings, word_to_idx)

    Raises:
        ModelLoadError: If the model cannot be loaded from MinIO
    """
    pd.set_option("mode.chained_assignment", None)
    np.random.seed(random_seed)

    minio_client = get_minio_client()
    model_name = "word-vector-embeddings.model"

    try:
        response = minio_client.get_object("data", model_name)
        model_bytes = response.read()

        temp_path = Path("/tmp") / model_name
        with open(temp_path, "wb") as f:
            f.write(model_bytes)

        w2v = gensim.models.Word2Vec.load(str(temp_path))
        temp_path.unlink()

        vocab = w2v.wv.index_to_key
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        embeddings_array = np.array([w2v.wv[word] for word in vocab])
        embeddings = torch.tensor(embeddings_array, dtype=torch.float32)

        return vocab, embeddings, word_to_idx

    except S3Error as e:
        raise ModelLoadError(
            f"Failed to load word2vec model from MinIO: {str(e)}. "
            "Please ensure the model file exists in the MinIO 'data' bucket."
        ) from e
    except Exception as e:
        raise ModelLoadError(
            f"Unexpected error loading word2vec model: {str(e)}"
        ) from e
    finally:
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    try:
        vocab, embeddings, word_to_idx = load_word2vec()
        print(f"Successfully loaded model. Embedding shape: {embeddings.shape}")
    except ModelLoadError as e:
        print(f"Error: {str(e)}")
