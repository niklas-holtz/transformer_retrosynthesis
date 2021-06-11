import math

import numpy as np
import tensorflow as tf
from rdkit import Chem

from ..Transformer import Transformer


class BeamSearchTranslator:

    def __init__(self, model: Transformer):
        self.model = model

    def predict(self, sequence, tk, max_length=160, beam_size=5, minimum_predictions=5, validity_check=True, max_false_predictions=10):
        """
        :param max_false_predictions:
        :param validity_check:
        :param minimum_predictions: minimum amount of predictions to return
        :param sequence: the sequence to be translated
        :param tk: the tokenizer
        :param max_length: the maximum length of the search
        :param beam_size: the size of the beam
        :return: the translated sequence
        """
        # Tokenize the input
        inp_sequence = tk.tokenize(sequence)
        inp_sequence = tf.convert_to_tensor(inp_sequence, np.int64)
        inp_sequence = tf.expand_dims(inp_sequence, 0)

        # Tokenize the start and end of sequence tokens
        sos_token = tf.convert_to_tensor(tk.get_sos_token(), np.int64)
        eos_token = tk.get_eos_token()

        # Initial output
        output = tf.convert_to_tensor([sos_token], np.int64)
        output = tf.expand_dims(output, 0)

        # Nodes with finished sequences
        fin_nodes = []

        # Count the false smiles predictions
        false_predictions = 0

        print('> Starting predictions ...')
        beam = Beam(self.model, inp_sequence, output, beam_size)

        # Start Prediction
        for i in range(max_length):
            # print('> Prediction number ', i)
            beam.next()

            for node in beam.nodes[:]:  # [:] creates a copy
                node_out = node.current_output.numpy()[0]

                # Check if one node created an eos token
                if node_out[-1] == eos_token:
                    beam.nodes.remove(node)
                    # Check if the node created a non valid string
                    node_smiles = tk.detokenize(node_out)

                    if validity_check and false_predictions < max_false_predictions:
                        if Chem.MolFromSmiles(node_smiles) is not None:
                            # Save the nodes and remove them from the original list and continue searching with other
                            # nodes
                            fin_nodes.append(node)
                        else:
                            # Increase the the false predictions amount
                            false_predictions += 1
                            # Look for another node
                    else:
                        fin_nodes.append(node)

            # Stop if at least n completed nodes have been found
            if len(fin_nodes) >= minimum_predictions:
                break

        # Add a few promising sequences if there haven't been found enough so far
        if len(fin_nodes) < minimum_predictions:
            diff = minimum_predictions - len(fin_nodes)
            if diff < len(beam.nodes):
                fin_nodes += beam.nodes[:(len(beam.nodes) - diff) * -1]
            else:
                fin_nodes += beam.nodes

        # Normalization
        for node in fin_nodes:
            node_out = node.current_output.numpy()[0]
            node.score *= 1 / len(node_out)

        # Sort again after normalization
        fin_nodes.sort(key=lambda x: x.score, reverse=True)

        # print('> Normlization ... ')
        # print_token_predictions(fin_nodes, tk)

        # Output the best one
        best_token_seq = fin_nodes[0].current_output.numpy()[0]
        text = tk.detokenize(best_token_seq)
        # text = best_token_seq

        # Output the rest
        all_token_seq = [tk.detokenize(token.current_output.numpy()[0]) for token in fin_nodes]
        # all_token_seq = [token.current_output.numpy()[0] for token in fin_nodes]

        print('> Prediction finished ... ')
        return text, best_token_seq, all_token_seq


def print_token_predictions(nodes, tk):
    for node in nodes:
        output = node.current_output
        output = output.numpy()[0]
        output = tk.detokenize(output)
        print(output, " > score = ", node.score)


def calc_scores(predictions):
    scores = []
    for _, (probability, token) in enumerate(predictions):
        score = math.log10(probability)
        scores.append(tuple([token, score]))
    return scores


def get_output_tensor(sequence):
    output = tf.convert_to_tensor([sequence], np.int64)
    output = tf.expand_dims(output, 0)
    return output


class Beam:

    def __init__(self, model, inp_sequence, start_output, beam_size):
        self.model = model
        self.inp_sequence = inp_sequence
        self.nodes = [SequenceNode(start_output, 0)]
        self.beam_size = beam_size

    def next(self):
        new_nodes = []
        for i, token in enumerate(self.nodes):
            predictions, _ = self.model(self.inp_sequence, token.current_output, False)
            # Select all tokens from the seq_len dimension
            predictions = predictions[:, -1:, :]

            # Apply softmax
            softmax = tf.keras.layers.Softmax()
            predictions = softmax(predictions)
            # Get the best predictions according to the beam size
            best_predictions = sorted([(x, i) for (i, x) in enumerate(predictions.numpy()[0][0])], reverse=True)[
                               :self.beam_size]

            # Calculate their scores
            best_predictions = calc_scores(
                best_predictions)  # [(23, -0.3760484563819582), (24, -0.6550012336733987)]
            for _, prediction in enumerate(best_predictions):
                # Concatenate the output with the new one
                output = get_output_tensor(prediction[0])
                output = tf.concat([token.current_output, output],
                                   axis=-1)  # tf.Tensor([[ 1 23]], shape=(1, 2), dtype=int64)
                # Add the score of this prediction to the existing one
                score = prediction[1] + token.score
                # Create new token for this output
                new_nodes.append(SequenceNode(output, score))

        # Sort the nodes according to their score and pick the best
        self.nodes = sorted(new_nodes, key=lambda x: x.score, reverse=True)[:self.beam_size]
        # print_token_predictions(self.nodes, self.tk)


class SequenceNode:
    def __init__(self, current_output, score):
        self.current_output = current_output
        self.score = score
