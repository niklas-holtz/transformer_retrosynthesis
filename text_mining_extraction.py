"""
This script was used to extract all chemical reactions from the text mining dataset and bundle them into a single file.
You can find the text mining dataset in text_mining/grants.
Duplicate reactions are automatically removed and incorrect SMILES strings are sorted out.

The Ph.D. thesis for the extraction of chemical structures and reactions from the literature can be found here:
https://www.repository.cam.ac.uk/handle/1810/244727
"""

import os
from os import walk
import xml.etree.ElementTree as ET
from rdkit import Chem
import model as trans
from role_asignment import identifyReactants
from rdkit.Chem import RDConfig, AllChem

# Initialize a Tokenizer in order to only keep reactions that can be tokenized
tk = trans.SmilesTokenizer()

# Collect all reactions from all files inside the directory
path = "text_mining/grants"
sub_folders = [f.path for f in os.scandir(path) if f.is_dir()]
files = []
for folder in sub_folders:
    for (dir_path, dir_names, filenames) in walk(folder):
        for name in filenames:
            files.append(dir_path + "/" + name)
print("Loading a total of " + str(len(files)) + " files.")


def assign_roles(reaction_smiles):
    """
    Assign the roles of each part of a reaction in order to get the most probable one.
    This is part of the contribution package of rdkit and can be found under the following link
    https://github.com/rdkit/rdkit/tree/d20077a089ca9e79b72a41934ac1d74a19bec3c3/Contrib/RxnRoleAssignment
    :param reaction_smiles: the chemical reaction as a smiles string
    :return: the adapted reaction as a smiles string
    """
    rxn = AllChem.ReactionFromSmarts(reaction_smiles, useSmiles=True)
    res = identifyReactants.identifyReactants(rxn, output=False)
    # Reactants identified by FP-based method
    reactants_idx = res[0]
    # Return if the role assignment wasn't successful
    if len(reactants_idx) < 1 or len(reactants_idx[0]) < 1:
        return reaction_smiles

    reactants_idx = reactants_idx[0]
    tmp = reaction_smiles.split(' >> ')
    reactants = tmp[0]
    prod = tmp[1]
    new_reactants = [r for i, r in enumerate(reactants.split('.')) if i in reactants_idx]

    print('Original: ' + reaction_smiles)
    fp_reaction = '.'.join(new_reactants) + ' >> ' + prod
    print('Adapted: ' + fp_reaction)

    return fp_reaction


def collect_smiles_by_tag(collection, tag, child):
    """
    This method collects the SMILES string for a specific XML tag (e.g. ProductList).
    :param collection: the collection where the found smiles string should be stored
    :param tag: the tag to be searched for
    :param child: the current xml-child of a tree
    :return: None (Call by Reference due to mutable lists)
    """
    # Check if the current child has the desired tag
    if tag not in child.tag:
        return

    for elem in child.iter():
        if 'identifier' in elem.tag and elem.attrib['dictRef'] == 'cml:smiles':
            elem_smiles = elem.attrib['value']
            # Check if its a valid smiles string and its tokenizable
            if Chem.CanonSmiles(elem_smiles) and tk.tokenize(elem_smiles).any():
                collection.append(elem_smiles)


# Open the new dataset file and write each reaction into it
with open("data/full_dataset_adapted.smi", 'w') as dest_file:
    skipped_files = 0
    reactions = []
    for i, file in enumerate(files):
        tree = ET.parse(file)
        root = tree.getroot()
        for reaction in root:
            try:
                s_products = []
                s_reagents = []
                s_reactants = []
                for child in reaction:
                    # Collect all products
                    collect_smiles_by_tag(s_products, 'productList', child)
                    # Collect all reactants
                    collect_smiles_by_tag(s_reactants, 'reactantList', child)
                    # Collect all reagents
                    collect_smiles_by_tag(s_reactants, 'spectatorList', child)

                # Combine all products, reactants and reagents of the reaction to a single line of SMILES-Code
                synthesis_final = '.'.join(s_reactants) + '.'.join(s_reagents) + ' >> ' + '.'.join(s_products)

                # Reaction Role Assignment
                synthesis_final = assign_roles(synthesis_final)

                # Form a retrosynthesis string by exchanging the part before the arrow (>>) with the part after it
                tmp = synthesis_final.split(' >> ')
                retro_final = tmp[1] + ' >> ' + tmp[0]

                print('Retro: ' + retro_final)
                reactions.append(retro_final)
                print(len(reactions))
            except Exception as error:
                # Skip the reaction
                skipped_files += 1
                continue

    # Remove all duplicates by using a dictionary
    print("Removing duplicates ... ")
    reactions = list(dict.fromkeys(reactions))
    print(len(reactions))
    print("Saving to file ... ")
    for entry in reactions:
        dest_file.write(entry + "\n")
    print("Skipped files .. " + str(skipped_files))
