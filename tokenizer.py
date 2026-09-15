class Tokenizer:
    def __init__(self, text):
        # Find every unique character
        self.chars = sorted(set(text))

        # Character → number
        self.char_to_id = {
            char: i for i, char in enumerate(self.chars)
        }

        # Number → character
        self.id_to_char = {
            i: char for i, char in enumerate(self.chars)
        }

        self.vocab_size = len(self.chars)

    def encode(self, text):
        """Convert text into token IDs."""
        return [self.char_to_id[c] for c in text]

    def decode(self, tokens):
        """Convert token IDs back into text."""
        return ''.join(self.id_to_char[i] for i in tokens)
