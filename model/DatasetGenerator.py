import io

import numpy as np
import tensorflow as tf


class DatasetGenerator:
    """
    This generator allows to create a dataset for a given path and encode it using a tokenizer.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    # Tokenize the data
    def preprocess_entry(self, w):
        return self.tokenizer.tokenize(w)

    # Load dataset
    def load_data_from_file(self, path, num_entries):
        # Load the lines from the file and separate them
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        print('Creating dataset for ' + str(num_entries) + ' out of ' + str(
            len(lines)) + ' found entries of the document.')
        # Preprocess data, split the entries and create a zip object
        word_pairs = [[self.preprocess_entry(w) for w in l.split(' >> ')[0:2]] for l in lines[:num_entries]]
        return zip(*word_pairs)

    def gen_from_file(self, path, num_entries, batch_size=64, padding_value=0):
        # Load data from path
        products, reactants = self.load_data_from_file(path, num_entries)

        # Add padding
        combined = np.concatenate((products, reactants))
        # value = self.tokenizer.char_to_ix['[nop]']
        combined = tf.keras.preprocessing.sequence.pad_sequences(combined, value=padding_value, padding='post',
                                                                 dtype='int64')
        products, reactants = np.split(combined, 2)

        print(products[-1])
        print(reactants[-1])

        # Define a dataset object
        dataset = tf.data.Dataset.from_tensor_slices((products, reactants))
        dataset = dataset.shuffle(len(products), reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
