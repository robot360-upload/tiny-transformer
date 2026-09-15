from tokenizer import Tokenizer
from embedding import Embedding
from positional_encoding import PositionalEncoding
from attention import SelfAttention


text = "hello world"

tokenizer = Tokenizer(text)

embedding = Embedding(
    tokenizer.vocab_size,
    16
)

tokens = tokenizer.encode("hello")

x = embedding.forward(tokens)

position = PositionalEncoding(
    max_length=32,
    embedding_dim=16
)

x = position.forward(x)

attention = SelfAttention(16)

output, weights = attention.forward(x)

print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)

print("\nAttention weights:")
print(weights)

print("\nRow sums:")
print(weights.sum(axis=1))
