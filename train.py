from tokenizer import Tokenizer
from embedding import Embedding
from positional_encoding import PositionalEncoding
from attention import MultiHeadCausalAttention


# Training text
text = "hello world"

# Tokenizer
tokenizer = Tokenizer(text)

print("Vocabulary:", tokenizer.chars)
print("Vocabulary size:", tokenizer.vocab_size)


# Encode text
tokens = tokenizer.encode("hello")

print("\nTokens:", tokens)


# Embedding
embedding_dim = 16
num_heads = 4

embedding = Embedding(
    tokenizer.vocab_size,
    embedding_dim
)

x = embedding.forward(tokens)

print("\nEmbedding shape:", x.shape)


# Positional encoding
position = PositionalEncoding(
    max_length=32,
    embedding_dim=embedding_dim
)

x = position.forward(x)

print("After positional encoding:", x.shape)


# Self-attention
attention = MultiHeadCausalAttention(
    embedding_dim,
    num_heads
)

output, weights = attention.forward(x)

print("Attention output:", output.shape)

print("\nAttention weights:")
print(weights)

print("\nRow sums:")
print(weights.sum(axis=2))

print("\nEverything works!")
