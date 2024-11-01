import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import io
from datetime import datetime
import logging

from app.embedding_utils import str_to_tokens

logger = logging.getLogger(__name__)


def create_training_triplet(
    query: str, relevant_doc: str, training_df: pd.DataFrame
) -> dict:
    """Create a training triplet from query-doc pair and random irrelevant doc"""
    random_index = np.random.randint(0, len(training_df))
    irrelevant_doc = training_df.iloc[random_index]["doc_relevant"]

    return {
        "query": query,
        "doc_relevant": relevant_doc,
        "doc_irrelevant": irrelevant_doc,
    }


def prepare_training_data(logs: list, training_df: pd.DataFrame) -> pd.DataFrame:
    """Convert logs to training triplets"""
    training_triplets = [
        create_training_triplet(log[2], log[3], training_df) for log in logs
    ]
    return pd.DataFrame(training_triplets)


def tokenize_data(df: pd.DataFrame, word_to_idx: dict) -> pd.DataFrame:
    """Tokenize the training data"""
    df["query_tokens"] = df["query"].apply(lambda x: str_to_tokens(x, word_to_idx))
    df["doc_rel_tokens"] = df["doc_relevant"].apply(
        lambda x: str_to_tokens(x, word_to_idx)
    )
    df["doc_irr_tokens"] = df["doc_irrelevant"].apply(
        lambda x: str_to_tokens(x, word_to_idx)
    )
    return df


def create_dataloader(dataset, batch_size: int = 32) -> DataLoader:
    """Create a DataLoader from the training data"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def save_model(model: nn.Module, minio_client, bucket: str):
    """Save model to MinIO"""
    # Save current state
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    minio_client.put_object(
        bucket,
        f"model_weights/model_{timestamp}.pth",
        buffer,
        buffer.getbuffer().nbytes,
    )

    # Save as latest
    buffer.seek(0)
    minio_client.put_object(
        bucket, "two_tower_state_dict.pth", buffer, buffer.getbuffer().nbytes
    )
    logger.info(f"Saved model with timestamp {timestamp}")


def fine_tune_model(model, dataloader, settings: dict) -> None:
    """Fine-tune the model with the provided dataloader"""
    optimizer = optim.Adam(model.parameters(), lr=settings["LEARNING_RATE"])
    model.train()

    for epoch in range(settings["NUM_EPOCHS"]):
        total_loss = 0
        for batch_rel, batch_irr, batch_query in dataloader:
            # Create masks
            rel_mask = (batch_rel != 0).float()
            irr_mask = (batch_irr != 0).float()
            query_mask = (batch_query != 0).float()

            # Forward pass
            similarity_rel = model(
                batch_rel, batch_query, doc_mask=rel_mask, query_mask=query_mask
            )
            similarity_irr = model(
                batch_irr, batch_query, doc_mask=irr_mask, query_mask=query_mask
            )

            # Calculate loss with higher weight for reinforcement data
            loss = model.margin - similarity_rel + similarity_irr
            loss = torch.relu(loss).mean() * settings["REINFORCEMENT_WEIGHT"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{settings['NUM_EPOCHS']}, Loss: {avg_loss:.4f}")

    return model


# Training settings
SETTINGS = {
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-5,
    "NUM_EPOCHS": 3,
    "PROJECTION_DIM": 64,
    "MARGIN": 0.5,
    "REINFORCEMENT_WEIGHT": 2.0,
}

if __name__ == "__main__":
    # This would be called from main.py with the logs and minio_client
    fine_tune_model(logs, minio_client, SETTINGS)
