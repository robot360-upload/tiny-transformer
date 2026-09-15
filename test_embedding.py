from tokenizer import Tokenizer
from embedding import Embedding

text = "hello world"

tokenizer = Tokenizer(text)

embedding = Embedding(
    tokenizer.vocab_size,
    16
)

tokens = tokenizer.encode("hello")

vectors = embedding.forward(tokens)

print("Tokens:")
print(tokens)

print("\nEmbedding shape:")
print(vectors.shape)

print("\nEmbedding vectors:")
print(vectors)
