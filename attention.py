import numpy as np


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class MultiHeadCausalAttention:
    def __init__(self, embedding_dim, num_heads):
        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        scale = 0.02

        self.W_Q = np.random.randn(
            embedding_dim, embedding_dim
        ) * scale

        self.W_K = np.random.randn(
            embedding_dim, embedding_dim
        ) * scale

        self.W_V = np.random.randn(
            embedding_dim, embedding_dim
        ) * scale

        self.W_O = np.random.randn(
            embedding_dim, embedding_dim
        ) * scale

    def forward(self, x):
        sequence_length = x.shape[0]

        # Create Q, K and V
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        # Split into heads
        Q = Q.reshape(
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        K = K.reshape(
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        V = V.reshape(
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        # Move heads to first dimension
        Q = Q.transpose(1, 0, 2)
        K = K.transpose(1, 0, 2)
        V = V.transpose(1, 0, 2)

        # Attention scores
        scores = (
            Q @ K.transpose(0, 2, 1)
        ) / np.sqrt(self.head_dim)

        # Causal mask
        mask = np.tril(
            np.ones(
                (sequence_length, sequence_length)
            )
        )

        scores = np.where(
            mask == 1,
            scores,
            -1e9
        )

        # Convert scores to probabilities
        weights = softmax(scores)

        # Weighted values
        head_outputs = weights @ V

        # Combine heads
        head_outputs = head_outputs.transpose(1, 0, 2)

        combined = head_outputs.reshape(
            sequence_length,
            self.embedding_dim
        )

        # Output projection
        output = combined @ self.W_O

        return output, weights