"""
This script was used to create combined dataset that includes the "retrosynthesis-all" and "full-dataset-apdated"
datasets with a total length of 225165 reactions.
"""
import io
from rdkit import Chem
import random

# Load the first dataset
all_path = "data/retrosynthesis-all.smi"
all_lines = io.open(all_path, encoding='UTF-8').read().strip().split('\n')

# Load the extended dataset
extended_path = "data/full-dataset-adapted.smi"
extended_lines = io.open(extended_path, encoding='UTF-8').read().strip().split('\n')
# Shuffle the array to have an even distribution of the reactions
random.shuffle(extended_lines)

max_lines = 225165
new_file = "data/retrosynthesis-combined.smi"
with open(new_file, 'w') as file:
    # Write all files from the original one into the file
    line_counter = len(all_lines)
    for line in all_lines:
        file.write(line + "\n")

    # Create an array that holds all products as canonical smiles strings
    canon_data = [Chem.CanonSmiles(line.split(' >> ')[0]) for line in all_lines]

    # Append each line of the extended dataset if its not already in the first one
    for line in extended_lines:
        if line_counter >= max_lines:
            break
        print("Checking reaction .. " + str(line_counter))
        prod = Chem.CanonSmiles(line.split(' >> ')[0])
        if prod not in canon_data:
            line_counter += 1
            file.write(line + "\n")
        else:
            print("Skipping reaction ..")
