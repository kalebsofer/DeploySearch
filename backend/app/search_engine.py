import torch
import torch.nn as nn
import pandas as pd
import faiss
import io
from minio import Minio
import logging
from typing import Optional
from .config.settings import get_settings
from .models.core import TwoTowerModel
from .utils.preprocess_str import str_to_tokens
from .utils.load_data import load_word2vec

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

settings = get_settings()

# Model constants
FREEZE_EMBEDDINGS = True
PROJECTION_DIM = 64
MARGIN = 0.5


class SearchEngine:
    def __init__(self):
        self._initialize_minio_client()
        self._initialize_resources()
        self.current_model_timestamp: Optional[str] = None
        logger.info("SearchEngine initialized")

    def _initialize_minio_client(self):
        self.minio_client = Minio(
            settings.MINIO_URL,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )

    def _get_file_from_minio(self, bucket: str, object_name: str) -> bytes:
        try:
            response = self.minio_client.get_object(bucket, object_name)
            return response.read()
        except Exception as e:
            raise Exception(f"Error loading {object_name} from MinIO: {str(e)}")

    def _get_latest_model_timestamp(self) -> Optional[str]:
        """Get the timestamp of the latest model from MinIO"""
        try:
            # List all objects in the model_weights directory
            objects = list(
                self.minio_client.list_objects("data", prefix="model_weights/model_")
            )

            if not objects:
                logger.info("No timestamped models found, will use default model")
                return None

            # Get the latest timestamp
            timestamps = [
                obj.object_name.split("model_")[1].split(".pth")[0] for obj in objects
            ]

            # Sort timestamps in descending order
            latest = sorted(timestamps, reverse=True)[0]
            logger.info(f"Found latest model timestamp: {latest}")
            return latest

        except Exception as e:
            logger.error(f"Error getting latest model timestamp: {e}")
            return None

    def _load_model_weights(self) -> None:
        """Load model weights from MinIO"""
        try:
            latest_timestamp = self._get_latest_model_timestamp()

            if latest_timestamp:
                # Try to load timestamped model
                try:
                    model_bytes = self._get_file_from_minio(
                        "data", f"model_weights/model_{latest_timestamp}.pth"
                    )
                    self.model.load_state_dict(
                        torch.load(
                            io.BytesIO(model_bytes),
                            weights_only=True,
                            map_location=torch.device("cpu"),
                        )
                    )
                    self.current_model_timestamp = latest_timestamp
                    logger.info(
                        f"Successfully loaded model with timestamp: {latest_timestamp}"
                    )
                    return
                except Exception as e:
                    logger.error(f"Failed to load timestamped model: {e}")
                    # Fall through to default model

            # Load default model if no timestamp or timestamped model failed
            logger.info("Loading default model (two_tower_state_dict.pth)")
            model_bytes = self._get_file_from_minio("data", "two_tower_state_dict.pth")
            self.model.load_state_dict(
                torch.load(
                    io.BytesIO(model_bytes),
                    weights_only=True,
                    map_location=torch.device("cpu"),
                )
            )
            self.current_model_timestamp = None
            logger.info("Successfully loaded default model")

        except Exception as e:
            logger.error(f"Error loading any model weights: {e}")
            raise Exception("Failed to load both timestamped and default models")

    def _initialize_resources(self):
        """Initialize model and resources"""
        try:
            # Load word vectors and initialize embedding layer
            self.vocab, embeddings, self.word_to_idx = load_word2vec()
            self.embedding_layer = nn.Embedding.from_pretrained(
                embeddings, freeze=FREEZE_EMBEDDINGS
            )

            # Initialize model architecture
            self.model = TwoTowerModel(
                embedding_dim=self.embedding_dim,
                embedding_layer=self.embedding_layer,
                projection_dim=PROJECTION_DIM,
                margin=MARGIN,
            )

            # Load latest model weights
            self._load_model_weights()
            self.model.eval()

            # Load other resources (FAISS index, etc.)
            # ... existing resource loading code ...

        except Exception as e:
            raise Exception(f"Error initializing search engine: {str(e)}")

    def check_for_model_updates(self) -> None:
        """Check and load new model weights if available"""
        self._load_model_weights()

    def _get_nearest_neighbors(self, query: str, k: int = 10):
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
