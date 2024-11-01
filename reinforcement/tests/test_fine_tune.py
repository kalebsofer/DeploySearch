import unittest
import pandas as pd
import torch
import numpy as np
from minio import Minio
import os
from app.config.settings import get_settings
from app.embedding_utils import EmbeddingProcessor
from app.train import fine_tune_model, create_dataloader, SETTINGS
import logging
from datetime import datetime
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFineTuning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with real MinIO client"""
        settings = get_settings()

        # Initialize MinIO client
        cls.minio_client = Minio(
            f"{settings.MINIO_HOST}:{settings.MINIO_PORT}",
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,  # Development environment
        )

        # Ensure bucket exists
        bucket_name = settings.MINIO_BUCKET
        try:
            if not cls.minio_client.bucket_exists(bucket_name):
                cls.minio_client.make_bucket(bucket_name)
                logger.info(f"Created bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Error checking/creating bucket: {str(e)}")
            raise

        # Initialize embedding processor with real MinIO client
        try:
            cls.embedding_processor = EmbeddingProcessor(cls.minio_client)
            logger.info("Successfully initialized EmbeddingProcessor")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingProcessor: {str(e)}")
            raise

    def test_fine_tuning(self):
        """Test the fine-tuning process with real data"""
        try:
            # Create base training data
            train_data = [
                {
                    "query": "how to deploy docker",
                    "doc_relevant": "Docker is a platform for developing, shipping, and running applications in containers. Containers package up code and all its dependencies.",
                    "doc_irrelevant": "Python is a high-level programming language known for its simple syntax and readability.",
                },
                {
                    "query": "python async programming",
                    "doc_relevant": "Asynchronous programming in Python allows you to write concurrent code using async and await keywords.",
                    "doc_irrelevant": "Docker containers provide a way to package applications with all their dependencies.",
                },
                {
                    "query": "kubernetes deployment",
                    "doc_relevant": "Kubernetes is a container orchestration platform that automates the deployment and scaling of containerized applications.",
                    "doc_irrelevant": "Git is a distributed version control system for tracking changes in source code.",
                },
            ]

            # Create DataFrame and duplicate rows 10 times for reinforcement
            train_df = pd.DataFrame(train_data)
            train_df = pd.concat([train_df] * 10, ignore_index=True)
            logger.info(
                f"Training data size after duplication: {len(train_df)} examples"
            )

            # Tokenize data
            train_df["query_tokens"] = train_df["query"].apply(
                lambda x: self.embedding_processor.str_to_tokens(x)
            )
            train_df["doc_rel_tokens"] = train_df["doc_relevant"].apply(
                lambda x: self.embedding_processor.str_to_tokens(x)
            )
            train_df["doc_irr_tokens"] = train_df["doc_irrelevant"].apply(
                lambda x: self.embedding_processor.str_to_tokens(x)
            )

            # Create dataset and dataloader
            dataset = self.embedding_processor.create_dataset(train_df)
            dataloader = create_dataloader(
                dataset, batch_size=4
            )  # Increased batch size

            # Save initial model state
            initial_state = {
                name: param.clone()
                for name, param in self.embedding_processor.model.named_parameters()
            }

            # Fine-tune model
            test_settings = {**SETTINGS, "NUM_EPOCHS": 1}
            fine_tuned_model = fine_tune_model(
                self.embedding_processor.model, dataloader, test_settings
            )

            # Verify model weights have changed
            for name, param in fine_tuned_model.named_parameters():
                if param.requires_grad:
                    self.assertFalse(
                        torch.allclose(param, initial_state[name]),
                        f"Parameters {name} did not change during fine-tuning",
                    )

            # Save fine-tuned model to MinIO test folder
            model_buffer = io.BytesIO()
            torch.save(fine_tuned_model.state_dict(), model_buffer)
            model_buffer.seek(0)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_model_path = f"test_models/model_{timestamp}.pth"

            self.minio_client.put_object(
                settings.MINIO_BUCKET,
                test_model_path,
                model_buffer,
                length=len(model_buffer.getvalue()),
            )

            logger.info(f"Test model saved to MinIO: {test_model_path}")
            logger.info("Fine-tuning test completed successfully")

        except Exception as e:
            logger.error(f"Test failed with error: {str(e)}")
            raise


if __name__ == "__main__":
    unittest.main()
