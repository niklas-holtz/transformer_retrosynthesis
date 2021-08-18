import numpy as np


class SmilesTokenizer:
    #chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTUVXYZ[\\]abcdefgilmnoprstuy$"
    chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"
    vocab_size = len(chars)

    def __init__(self):
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def get_vocab_size(self):
        return self.vocab_size

    def get_sos_token(self):
        return self.char_to_ix['^']

    def get_eos_token(self):
        return self.char_to_ix['$']

    def tokenize(self, input):
        # Add start and end tag
        input = '^' + input + '$'
        # Collect all inputs and translate them to our vocabulary
        result = np.zeros(shape=(len(input)), dtype=np.int64)
        for i, char in enumerate(input.strip()):
            result[i] = self.char_to_ix[char]
        return result

    def detokenize(self, input):
        result = str()
        for i, char in enumerate(input):
            result += self.ix_to_char[char]

        # Remove start and end tag
        if result.startswith('^') and result.endswith('$'):
            result = result[1:-1]

        return result
