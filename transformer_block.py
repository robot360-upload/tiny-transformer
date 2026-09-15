import numpy as np

from attention import MultiHeadCausalAttention


class LayerNorm:
    def __init__(self, embedding_dim, epsilon=1e-5):
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon

        self.gamma = np.ones(embedding_dim)
        self.beta = np.zeros(embedding_dim)

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        normalized = (
            (x - mean)
            / np.sqrt(variance + self.epsilon)
        )

        return self.gamma * normalized + self.beta


class FeedForward:
    def __init__(self, embedding_dim, hidden_dim):
        scale = 0.02

        self.W1 = (
            np.random.randn(embedding_dim, hidden_dim)
            * scale
        )

        self.b1 = np.zeros(hidden_dim)

        self.W2 = (
            np.random.randn(hidden_dim, embedding_dim)
            * scale
        )

        self.b2 = np.zeros(embedding_dim)

    def forward(self, x):
        hidden = x @ self.W1 + self.b1

        # ReLU activation
        hidden = np.maximum(0, hidden)

        output = hidden @ self.W2 + self.b2

        return output


class TransformerBlock:
    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim
    ):
        self.attention = MultiHeadCausalAttention(
            embedding_dim,
            num_heads
        )

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim,
            hidden_dim
        )

    def forward(self, x):
        # Self-attention
        attention_output, weights = (
            self.attention.forward(x)
        )

        # First residual connection + LayerNorm
        x = self.norm1(
            x + attention_output
        )

        # Feed-forward network
        ff_output = self.feed_forward.forward(x)

        # Second residual connection + LayerNorm
        x = self.norm2(
            x + ff_output
        )

        return x, weights
