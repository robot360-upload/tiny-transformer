import numpy as np


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)

    exp_x = np.exp(x)

    return exp_x / np.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )


class OutputProjection:
    def __init__(self, embedding_dim, vocab_size):
        scale = 0.02

        self.W = (
            np.random.randn(
                embedding_dim,
                vocab_size
            ) * scale
        )

        self.b = np.zeros(vocab_size)

    def forward(self, x):
        logits = x @ self.W + self.b

        return logits

    def probabilities(self, x):
        logits = self.forward(x)

        return softmax(logits)
