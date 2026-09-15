import numpy as np


def softmax(x):
    # Subtract maximum for numerical stability
    x = x - np.max(x, axis=-1, keepdims=True)

    exp_x = np.exp(x)

    return exp_x / np.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )


class SelfAttention:

    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

        # Trainable matrices
        scale = 0.02

        self.W_Q = (
            np.random.randn(
                embedding_dim,
                embedding_dim
            ) * scale
        )

        self.W_K = (
            np.random.randn(
                embedding_dim,
                embedding_dim
            ) * scale
        )

        self.W_V = (
            np.random.randn(
                embedding_dim,
                embedding_dim
            ) * scale
        )

    def forward(self, x):

        # Create queries, keys and values
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        # Attention scores
        scores = (
            Q @ K.T
        ) / np.sqrt(self.embedding_dim)

        # Convert scores to probabilities
        weights = softmax(scores)

        # Weighted combination of values
        output = weights @ V

        return output, weights
