import io
import random

import selfies as sf
import rdkit.Chem as rkc
from rdkit import Chem


def to_mol(smiles):
    return rkc.MolFromSmiles(smiles)

def randomize_smiles(mol, random_type="restricted"):
    if not mol:
        return None

    if random_type == "unrestricted":
        return rkc.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    if random_type == "restricted":
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = rkc.RenumberAtoms(mol, newOrder=new_atom_order)
        return rkc.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
    raise ValueError("Type '{}' is not valid".format(random_type))


num_entries = None
path = "data/retrosynthesis-all.smi"

# Load the lines from the file and separate them
lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
print('Creating dataset for ' + str(num_entries) + ' out of ' + str(len(lines)) + ' found entries of the document.')
# Split the entries
word_pairs = [[w for w in l.split(' >> ')[0:2]] for l in lines[:num_entries]]


def gen_random(prod, reactant, amount):
    for i in range(amount):
        # Alternative form using selfies decode
        prod2 = randomize_smiles(to_mol(prod))

        attempts = 0
        fail = False
        while Chem.CanonSmiles(prod) != Chem.CanonSmiles(prod2):
            random.seed()
            prod2 = randomize_smiles(to_mol(prod))
            attempts += 1
            if attempts > 10:
                fail = True
                break

        if fail:
            print("fail!")
            continue

        reactant2 = randomize_smiles(to_mol(reactant))

        attempts = 0
        fail = False
        while Chem.CanonSmiles(reactant) != Chem.CanonSmiles(reactant2):
            random.seed()
            reactant2 = randomize_smiles(to_mol(reactant))
            attempts += 1
            if attempts > 10:
                fail = True
                break

        if fail:
            print("fail!")
            continue

        file.write(prod2 + " >> " + reactant2 + "\n")

with open("data/retrosynthesis-artificial_7.smi", 'w') as file:
    for i, pair in enumerate(word_pairs):
        prod = pair[0]
        reactant = pair[1]
        file.write(prod + " >> " + reactant + "\n")

        gen_random(prod, reactant, 8)


