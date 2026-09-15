import numpy as np


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)

    exp_x = np.exp(x)

    return exp_x / np.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )


class MultiHeadCausalAttention:

    def __init__(self, embedding_dim, num_heads):

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        scale = 0.02

        # One large matrix for each projection.
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

        # Output projection
        self.W_O = (
            np.random.randn(
                embedding_dim,
                embedding_dim
            ) * scale
        )

    def forward(self, x):

        sequence_length = x.shape[0]

        # --------------------------------
        # 1. Create Q, K and V
        # --------------------------------

        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        # --------------------------------
        # 2. Split into attention heads
        # --------------------------------

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

        # --------------------------------
        # 3. Move heads before sequence
        # --------------------------------

        Q = Q.transpose(1, 0, 2)
        K = K.transpose(1, 0, 2)
        V = V.transpose(1, 0, 2)

        # Shape:
        # (heads, sequence, head_dim)

        # --------------------------------
        # 4. Calculate attention scores
        # --------------------------------

        scores = (
            Q @ K.transpose(0, 2, 1)
        ) / np.sqrt(self.head_dim)

        # Shape:
        # (heads, sequence, sequence)

        # --------------------------------
        # 5. Causal mask
        # --------------------------------

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

        # --------------------------------
        # 6. Softmax
        # --------------------------------

        weights = softmax(scores)

        # --------------------------------
        # 7. Weighted values
        # --------------------------------

        head_outputs = weights @ V

        # --------------------------------
        # 8. Put heads back together
        # --------------------------------

        head_outputs = head_outputs.transpose(
            1, 0, 2
        )

        combined = head_outputs.reshape(
            sequence_length,
            self.embedding_dim
       