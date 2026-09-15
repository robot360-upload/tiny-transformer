import numpy as np


class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Trainable parameters
        self.weights = (
            np.random.randn(vocab_size, embedding_dim)
            * 0.02
        )

    def forward(self, tokens):
        """
        Convert token IDs into vectors.
        """
        return self.weights[tokens]
