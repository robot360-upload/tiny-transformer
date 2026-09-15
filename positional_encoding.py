import numpy as np


class PositionalEncoding:
    def __init__(self, max_length, embedding_dim):
        self.max_length = max_length
        self.embedding_dim = embedding_dim

        self.positions = np.zeros(
            (max_length, embedding_dim)
        )

        position = np.arange(max_length)[:, np.newaxis]

        div_term = np.exp(
            np.arange(0, embedding_dim, 2)
            * -(np.log(10000.0) / embedding_dim)
        )

        self.positions[:, 0::2] = np.sin(
            position * div_term
        )

        self.positions[:, 1::2] = np.cos(
            position * div_term
        )

    def forward(self, x):
        sequence_length = x.shape[0]

        return x + self.positions[:sequence_length]
