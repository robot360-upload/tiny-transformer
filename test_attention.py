from tokenizer import Tokenizer
from embedding import Embedding
from positional_encoding import PositionalEncoding
from attention import MultiHeadCausalAttention


text = "hello world"

tokenizer = Tokenizer(text)

embedding_dim = 16
num_heads = 4

embedding = Embedding(
    tokenizer.vocab_size,
    embedding_dim
)

tokens = tokenizer.encode("hello")

x = embedding.forward(tokens)

position = PositionalEncoding(
    max_length=32,
    embedding_dim=embedding_dim
)

x = position.forward(x)

attention = MultiHeadCausalAttention(
    embedding_dim,
    num_heads
)

output, weights = attention.forward(x)

print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)

print("\nAttention weight shape:")
print(weights.shape)

print("\nHead 0:")
print(weights[0])