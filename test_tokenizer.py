from tokenizer import Tokenizer

text = "hello world"

tokenizer = Tokenizer(text)

print("Vocabulary:", tokenizer.chars)
print("Vocabulary size:", tokenizer.vocab_size)

encoded = tokenizer.encode("hello")

print("Encoded:", encoded)

decoded = tokenizer.decode(encoded)

print("Decoded:", decoded)
