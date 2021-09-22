import io
import math

from rdkit import Chem
import requests
import json
import copy
import tensorflow as tf


def nodes_finished(nodes):
    for node in nodes:
        if node.is_finished():
            return True
    return False


class BeamAnalyser:

    def __init__(self, translator, transformer, dict_path):
        self.dict = {}
        self.translator = translator
        self.transformer = transformer
        self.init_dict(dict_path)

    def init_dict(self, path):
        print('Initializing dictionary ...')
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        for i, line in enumerate(lines):
            tmp = line.split(' >> ')
            self.dict[tmp[0]] = tmp[1]
        print('Dictionary has been initialized!')

    def is_available_on_molport(self, smiles):
        # Create a POST request using CURL
        headers = {
            'Content-type': 'application/json',
        }
        data = '{"API Key":"880d8343-8ui2-418c-9g7a-68b4e2e78c8b",' \
               '"Structure":"' + smiles + '",' \
                                          '"Search Type":"4",' \
                                          '"Maximum Search Time":60000,' \
                                          '"Maximum Result Count":1}'

        print("Sending Molport request ... ")
        try:
            response = requests.post('https://api.molport.com/api/chemical-search/search', headers=headers, data=data)
        except TimeoutError:
            return False

        print("Got Molport response ... ")
        data = json.loads(response.text)['Data']
        molecules = data['Molecules']
        print(molecules)
        return len(molecules) > 0

    def is_available_in_dict(self, smiles):
        found = smiles in self.dict
        if found:
            print("Found in dictionary ... " + smiles)
        return found

    def is_mol_available(self, smiles):
        return self.is_available_on_molport(smiles) or self.is_available_in_dict(smiles)

    def analyse(self, product, tokenizer, beam_size=6):
        global fail
        nodes = [ProductNode(product, 0)]

        iter = 1
        fail = False
        while not nodes_finished(nodes) and not fail:
            print("> Iteration: " + str(iter))

            new_nodes = []
            for node in nodes:
                for leave_idx, leave in enumerate(node.get_unfinished_leaves()):
                    token_decoded, all_token_scores, all_token_seq_decoded = self.translator.predict(leave.product,
                                                                                                     tokenizer,
                                                                                                     beam_size=beam_size,
                                                                                                     minimum_predictions=beam_size)
                    # Apply softmax to the scores
                    softmax = tf.keras.layers.Softmax()
                    scores = softmax(all_token_scores).numpy()
                    for prod_idx, prod in enumerate(all_token_seq_decoded):
                        # Create a new node for each product
                        new_node = copy.deepcopy(node)
                        # Find the same leave inside the copy by its index
                        new_node_leave = new_node.get_unfinished_leaves()[leave_idx]

                        # Skip wrong predictions
                        if not Chem.MolFromSmiles(prod):
                            continue

                        # Add all reactants as leaves to the new node
                        reactants = prod.split('.')

                        # Skip if its the same as the leave
                        if len(reactants) == 1 and Chem.CanonSmiles(prod) == Chem.CanonSmiles(leave.product):
                            continue

                        for prod_part in reactants:
                            if not Chem.MolFromSmiles(prod_part):
                                continue
                            canon_token = Chem.CanonSmiles(prod_part)
                            # Calculate the proportional score
                            score = scores[prod_idx] / len(reactants)
                            # Create a new ProductNode for the token
                            token_node = ProductNode(canon_token, score)
                            # Check if the molecule is available
                            token_node.finished = self.is_mol_available(canon_token)
                            # Add the node to the current leave
                            new_node_leave.add_reactant(token_node)
                        new_nodes.append(new_node)

            if len(new_nodes) < 1:
                fail = True
                break

            # Sort the nodes according to their score and pick the best
            nodes = sorted(new_nodes, key=lambda x: x.get_log_score(), reverse=True)[:beam_size]

            # Output nodes
            for node in nodes:
                print(node)
                print("Total score = " + str(node.get_log_score()))
            iter += 1

        print("Final nodes ... ")
        for node in nodes:
            print(node)

        for node in nodes:
            if node.is_finished():
                return node

        return None


class ProductNode:
    def __init__(self, product, score):
        self.product = product
        self.reactants = []
        self.finished = False
        self.score = score

    def is_finished(self):
        if len(self.reactants) < 1:
            return self.finished

        for child in self.reactants:
            if not child.is_finished():
                return False

        return True

    def get_unfinished_leaves(self):
        if len(self.reactants) < 1 and not self.finished:
            return [self]

        leaves = []
        for child in self.reactants:
            leaves += child.get_unfinished_leaves()

        return leaves

    def add_reactant(self, node):
        self.reactants.append(node)

    def get_log_score(self):
        return math.log10(self.get_score())

    def get_score(self):
        if len(self.reactants) < 1:
            return self.score

        score = self.score
        for child in self.reactants:
            score += child.get_score()

        return score

    def __str__(self, level=0):
        finished_str = lambda x: "(✓)" if x else ""
        ret = "\t" * level + repr(self.product) + ' (score = ' + str(self.score) + ') ' + finished_str(
            self.finished) + "\n"
        for child in self.reactants:
            ret += child.__str__(level + 1)
        return ret
