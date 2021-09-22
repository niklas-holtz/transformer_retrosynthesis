import io

import selfies as sf

from . import SmilesTokenizer


class SelfiesTokenizer(SmilesTokenizer):

    def __init__(self):
        super().__init__()
        self.alphabet = ['[nop]', '[^]', '.', '[C@expl]', '[/S@expl]', '[Br]', '[=PH2expl]', '[=PHexpl]', '[/I]',
                         '[B-expl]', '[C@Hexpl]', '[Branch1_2]', '[Expl=Ring2]', '[P]', '[Mg+expl]',
                         '[NH2+expl]', '[S@expl]', '[Expl=Ring1]', '[NH+expl]', '[/S]', '[Cl]', '[C@@expl]', '[Seexpl]',
                         '[=SHexpl]',
                         '[C@@Hexpl]', '[Branch1_3]', '[SiH2expl]', '[C-expl]', '[=P+expl]', '[P+expl]', '[S@@expl]',
                         '[Siexpl]', '[#C]',
                         '[PH4expl]', '[Ring1]', '[SnHexpl]', '[Branch1_1]', '[=P]', '[O-expl]', '[S+expl]',
                         '[BH-expl]', '[Mgexpl]',
                         '[/C@Hexpl]', '[\\N]', '[Cuexpl]', '[\\C]', '[=N+expl]', '[N+expl]', '[\\S@expl]', '[F]',
                         '[=N]', '[/S@@expl]',
                         '[\\B]', '[Ptexpl]', '[\\F]', '[NHexpl]', '[\\S]', '[PH2expl]', '[Pdexpl]', '[N]', '[/Br]',
                         '[\\S@@expl]', '[=C]',
                         '[B]', '[Snexpl]', '[/C]', '[#N+expl]', '[Zn+expl]', '[BH3-expl]', '[#N]', '[Cl+3expl]',
                         '[/Snexpl]', '[/N]',
                         '[\\O]', '[I]', '[Branch2_2]', '[/O]', '[\\Cl]', '[#C-expl]', '[/F]', '[/P]', '[C]',
                         '[OH-expl]', '[=O]', '[Feexpl]',
                         '[SiHexpl]', '[\\C@Hexpl]', '[NH3+expl]', '[=S@@expl]', '[O]', '[Cl-expl]', '[=Ptexpl]',
                         '[/N+expl]', '[SHexpl]',
                         '[Branch2_1]', '[Znexpl]', '[\\Br]', '[PHexpl]', '[\\C@@Hexpl]', '[/B]', '[Liexpl]',
                         '[NH4+expl]', '[/Cl]',
                         '[Branch2_3]', '[=S]', '[N-expl]', '[Ring2]', '[I+expl]', '[=N-expl]', '[S]', '[NH-expl]',
                         '[\\I]', '[Kexpl]',
                         '[S-expl]', '[/C@@Hexpl]', '[=S+expl]', '[Br-expl]', '[N@+expl]', '[$]']
        self.char_to_ix = {s: i for i, s in enumerate(self.alphabet)}
        self.ix_to_char = {i: s for i, s in enumerate(self.alphabet)}

    def gen_alphabet(self, path, num_entries):
        # Load the lines from the file and separate them
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        dataset = []
        print("found " + str(len(lines)))
        for idx, l in enumerate(lines[:num_entries]):
            print(idx)
            for w in l.split(' >> ')[0:2]:
                selfies = sf.encoder(w)
                print("appending ... " + selfies)
                dataset.append(selfies)
        # Append additional information to the alphabet
        self.alphabet = []
        self.alphabet.append('[nop]')
        self.alphabet.append('[^]')
        self.alphabet.append('.')
        self.alphabet += sf.get_alphabet_from_selfies(dataset)
        self.alphabet.append('[$]')
        return self.alphabet

    def load_alphabet_from_file(self, path):
        self.alphabet = []
        with open(path, 'r') as reader:
            line = reader.readline()
            while line != '':
                self.alphabet.append(line.rstrip())
                line = reader.readline()

        self.char_to_ix = {s: i for i, s in enumerate(self.alphabet)}
        self.ix_to_char = {i: s for i, s in enumerate(self.alphabet)}

    def tokenize(self, input):
        # Encode to selfies
        encoded_selfies = sf.encoder(input)
        # Add start and end tag
        encoded_selfies = '[^]' + encoded_selfies + '[$]'
        # Encode to label
        result = sf.selfies_to_encoding(encoded_selfies, vocab_stoi=self.char_to_ix, enc_type='label')
        return result

    def detokenize(self, input):
        result = sf.encoding_to_selfies(input, vocab_itos=self.ix_to_char, enc_type='label')

        # Remove all [nop] tokens
        result = result.replace('[nop]', '')

        if result.startswith('[^]'):
            result = result[3:]
        if result.endswith('[$]'):
            result = result[:-3]

        print("detokenizing ... " + result)
        return sf.decoder(result)

    def get_sos_token(self):
        return self.char_to_ix['[^]']

    def get_eos_token(self):
        return self.char_to_ix['[$]']

    def get_vocab_size(self):
        return len(self.alphabet)
