import psycopg2
from minio import Minio
import logging
import pandas as pd
import io
from app.embedding_utils import EmbeddingProcessor
from app.train import create_dataloader, fine_tune_model, save_model, SETTINGS
from app.config.settings import get_settings
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReinforcementService:
    def __init__(self):
        self.settings = get_settings()
        self._initialize_clients()
        logger.info("ReinforcementService initialized with settings")

    def _initialize_clients(self):
        """Initialize MinIO client and embedding processor"""
        try:
            self.minio_client = Minio(
                f"{self.settings.MINIO_HOST}:{self.settings.MINIO_PORT}",
                access_key=self.settings.MINIO_ACCESS_KEY,
                secret_key=self.settings.MINIO_SECRET_KEY,
                secure=False,
            )
            self.embedding_processor = EmbeddingProcessor(self.minio_client)
            logger.info("Successfully initialized MinIO client and embedding processor")
        except Exception as e:
            logger.error(f"Error initializing clients: {e}")
            raise

    def get_db_connection(self):
        return psycopg2.connect(
            dbname=self.settings.POSTGRES_DB,
            user=self.settings.POSTGRES_USER,
            password=self.settings.POSTGRES_PASSWORD,
            host="postgres",
            port="5432",
        )

    def fetch_feedback_logs(self, limit: int = 100):
        """Fetch the last N logs where feedback was received"""
        logger.info(f"Fetching last {limit} logs with feedback")

        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, session_id, query, selected_document, 
                           similarity_score, feedback_received, created_at
                    FROM search_logs
                    WHERE feedback_received = TRUE
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                logs = cur.fetchall()
                logger.info(f"Found {len(logs)} logs with feedback")
                return logs

    def create_training_triplet(
        self, query: str, relevant_doc: str, training_df: pd.DataFrame
    ) -> dict:
        """Create a training triplet from query-doc pair and random irrelevant doc"""
        random_index = np.random.randint(0, len(training_df))
        irrelevant_doc = training_df.iloc[random_index]["doc_relevant"]

        return {
            "query": query,
            "doc_relevant": relevant_doc,
            "doc_irrelevant": irrelevant_doc,
        }

    def prepare_training_data(self, logs: list) -> pd.DataFrame:
        """Convert logs to training triplets"""
        # Load training data from MinIO
        training_df = pd.read_parquet(
            io.BytesIO(
                self.minio_client.get_object(
                    "data", "training-with-tokens.parquet"
                ).read()
            )
        )

        training_triplets = [
            self.create_training_triplet(log[2], log[3], training_df) for log in logs
        ]
        return pd.DataFrame(training_triplets)

    def tokenize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tokenize the training data"""
        df["query_tokens"] = df["query"].apply(
            lambda x: self.embedding_processor.str_to_tokens(x)
        )
        df["doc_rel_tokens"] = df["doc_relevant"].apply(
            lambda x: self.embedding_processor.str_to_tokens(x)
        )
        df["doc_irr_tokens"] = df["doc_irrelevant"].apply(
            lambda x: self.embedding_processor.str_to_tokens(x)
        )
        return df

    def train(self, logs):
        """Main training function"""
        try:
            # Prepare training data
            train_df = self.prepare_training_data(logs)
            train_df = self.tokenize_data(train_df)

            logger.info(f"Prepared {len(train_df)} training examples")

            # Create dataset and dataloader
            dataset = self.embedding_processor.create_dataset(train_df)
            dataloader = create_dataloader(dataset, batch_size=SETTINGS["BATCH_SIZE"])

            # Fine-tune model
            model = fine_tune_model(
                self.embedding_processor.model, dataloader, SETTINGS
            )

            # Save model
            save_model(model, self.minio_client, "data")
            logger.info("Model training completed and saved successfully")

        except Exception as e:
            logger.error(f"Error in training process: {e}")
            raise

    def run_once(self):
        """Single execution of the training process"""
        try:
            logs = self.fetch_feedback_logs(limit=100)
            if logs:
                logger.info(f"Starting training process with {len(logs)} logs")
                self.train(logs)
                logger.info("Training completed successfully")
            else:
                logger.info("No feedback logs found for training")

        except Exception as e:
            logger.error(f"Error in training process: {e}")
            raise


if __name__ == "__main__":
    service = ReinforcementService()
    service.run_once()
