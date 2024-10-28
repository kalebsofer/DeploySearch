import torch
from .utils.preprocess_str import preprocess_query
from .utils.load_data import load_word2vec

def get_string_embedding(input_string, vocab, word_to_idx, embeddings):

    processed_words = preprocess_query(input_string)
    word_embeddings = []

    for word in processed_words:
        if word in word_to_idx:
            word_idx = word_to_idx[word]
            word_embedding = embeddings[word_idx]
            word_embeddings.append(word_embedding)

    if not word_embeddings:
        return torch.zeros(embeddings.shape[1])

    string_embedding = torch.stack(word_embeddings).mean(dim=0)
    return string_embedding

def preprocess(query: str):

    vocab, embeddings, word_to_idx = load_word2vec()

    embedding = get_string_embedding(query, vocab, word_to_idx, embeddings)
    return embedding

def str_to_tokens(s: str, word_to_idx: dict[str, int]) -> list[int]:
    split = preprocess_query(s)
    return [word_to_idx[word] for word in split if word in word_to_idx]


def main():
    query = "How to train a neural network"
    embedding = preprocess(query)
    print(f"Embedding shape: {embedding.shape}")
    print(f"First few values: {embedding[:5]}")

if __name__ == "__main__":  # Correct main check syntax
    main()
