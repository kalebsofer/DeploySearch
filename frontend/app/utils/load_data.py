import os
import torch
import gensim
import numpy as np
import pandas as pd
import gensim.downloader as api
from pathlib import Path

from .preprocess_str import preprocess_list as preprocess


def load_word2vec(random_seed=42, embeddings_path='./models/word-vector-embeddings.model', save_path='./models/word-vector-embeddings.model'):
    pd.set_option('mode.chained_assignment', None)  # Suppress SettingWithCopyWarning
    np.random.seed(random_seed)

    # Get the absolute path to the models directory
    base_dir = Path(__file__).parent.parent  # Go up two levels from utils to app
    models_dir = base_dir / "models"
    save_path = models_dir / "word-vector-embeddings.model"

    if save_path.exists():
        w2v = gensim.models.Word2Vec.load(str(save_path))
    else:
        # Train word2vec model
        raw_corpus = api.load('text8')
        corpus = [preprocess(doc) for doc in raw_corpus]
        w2v = gensim.models.Word2Vec(corpus, vector_size=100, window=5, min_count=3, workers=4)
        if save_path:
            w2v.save(str(save_path))

    # Load the word2vec model, extract embeddings, convert to torch tensor
    # w2v = gensim.models.Word2Vec.load('./word2vec/word2vec-gensim-text8-custom-preprocess.model')
    vocab = w2v.wv.index_to_key
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    embeddings_array = np.array([w2v.wv[word] for word in vocab])
    embeddings = torch.tensor(embeddings_array, dtype=torch.float32)
    return vocab, embeddings, word_to_idx


if __name__ == '__main__':
    vocab, embeddings, word_to_idx = load_word2vec()
    print(embeddings.shape)
