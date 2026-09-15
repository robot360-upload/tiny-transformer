import numpy as np

from tokenizer import Tokenizer
from embedding import Embedding
from positional_encoding import PositionalEncoding


text = "hello world"

tokenizer = Tokenizer(text)

embedding = Embedding(
    tokenizer.vocab_size,
    16
)

tokens = tokenizer.encode("hello")

x = embedding.forward(tokens)

print("Before positional encoding:")
print(x.shape)

position = PositionalEncoding(
    max_length=32,
    embedding_dim=16
)

x = position.forward(x)

print("After positional encoding:")
print(x.shape)

print("\nFirst token:")
print(x[0])
