"""
This script is used to generate randomized smiles for a given retrsonythesis dataset. It is strongly inspired by
the publication "Randomized SMILES strings improve the quality of molecular generative models" and and it basically
shuffles the atomic order of each molecule to create alternative SMILES representations.

A related script used for the publication can be found in the following repository:
https://github.com/undeadpixel/reinvent-randomized/blob/master/create_randomized_smiles.py
"""
import io
import random

import rdkit.Chem as rkc


# Creates the mol structure from a smiles string
def to_mol(smiles):
    return rkc.MolFromSmiles(smiles)


# Creates a randomized smiles string for a given mol structure by shuffling the atom order
def randomize_smiles(mol):
    if not mol:
        return None

    new_atom_order = list(range(mol.GetNumAtoms()))
    random.shuffle(new_atom_order)
    random_mol = rkc.RenumberAtoms(mol, newOrder=new_atom_order)
    return rkc.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)


# Load the lines from the file and separate them
num_entries = None
path = "data/retrosynthesis-all.smi"
lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
print('Creating dataset for ' + str(num_entries) + ' out of ' + str(len(lines)) + ' found entries of the document.')
# Split the entries
word_pairs = [[w for w in l.split(' >> ')[0:2]] for l in lines[:num_entries]]


# Generates a desired amount of random smiles strings for products and reactants
def gen_random(prod, reactant, amount):
    for i in range(amount):
        random.seed()
        new_prod = randomize_smiles(to_mol(prod))
        new_reactant = randomize_smiles(to_mol(reactant))
        file.write(new_prod + " >> " + new_reactant + "\n")


# Open the new file and write all randomized smiles into it
with open("data/retrosynthesis-artificial.smi", 'w') as file:
    for i, pair in enumerate(word_pairs):
        prod = pair[0]
        reactant = pair[1]
        file.write(prod + " >> " + reactant + "\n")
        gen_random(prod, reactant, 4)
