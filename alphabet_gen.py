"""
This script is used to generate an alphabet for a given dataset.
The alphabet is then saved in the folder "alphabet".
"""

import model as trans

path = 'data/full-dataset-adapted-canon.smi'

tk = trans.SelfiesTokenizer()
alphabet = tk.gen_alphabet(path, None)

with open('alphabets/retrosynthesis-full-canon-alphabet.txt', 'w') as writer:
    for selfies in alphabet:
        writer.write(selfies + '\n')


