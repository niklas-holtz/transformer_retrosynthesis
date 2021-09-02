"""
This script is used to generate an alphabet for a given dataset.
The alphabet is then saved in the folder "alphabet".
"""

import model as trans

path = 'data/retrosynthesis-artificial-2.smi'

tk = trans.SelfiesTokenizer()
alphabet = tk.gen_alphabet(path, None)

with open('alphabets/retrosynthesis-artificial-2-alphabet.txt', 'w') as writer:
    for selfies in alphabet:
        writer.write(selfies + '\n')


