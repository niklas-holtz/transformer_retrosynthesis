import io
import time
import numpy as np
import tensorflow as tf

from model.Tokenizer import Tokenizer
from model.Transformer import Transformer
from model.translators.BeamSearch import BeamSearchTranslator

# To show proper values when using numpy
np.set_printoptions(precision=3)
np.set_printoptions(suppress=False)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# Some hyper variables
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
EPOCHS = 10

# Tokenizer
tk = Tokenizer()

# Create the model
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tk.get_vocab_size(),
    target_vocab_size=tk.get_vocab_size(),
    pe_input=1000,
    pe_target=1000,
    rate=dropout_rate)

# Load the first line to initialize the transformer model
with io.open('data/retrosynthesis-train.smi') as data:
    line = data.read().strip().split('\n')[0].split(' >> ')
    line[0] = tk.tokenize(line[0])
    line[1] = tk.tokenize(line[1])
    line = tf.keras.preprocessing.sequence.pad_sequences(line, value=0, padding='post', dtype='int64', maxlen=199)
    inp, tar = np.split(line, 2)

    tar_inp = tar[:, :-1]
    predictions, _ = transformer(inp, tar_inp, False)

# Load saved trained_models
transformer.load_weights('trained_models/tr-1.h5')

# Create the translator
translator = BeamSearchTranslator(transformer)


def predict_smiles(smiles, expected=""):
    """
    Creates a prediction based on a single smiles string.
    :param smiles: the input smiles string
    :param expected: the expected output (ground truth)
    :return: the predicted smiles string (output)
    """
    translated_smiles, translated_tokens, all_tokens = translator.predict(smiles, tk)
    print("Input smiles: \t{}".format(smiles))
    print("Output smiles: \t{}".format(translated_smiles))
    print("Ground truth: \t{}".format(expected))
    print("All predictions: {}".format(all_tokens))
    return translated_smiles, translated_tokens, all_tokens


# Predict single smmiles
# predict_smiles("c1ccc(Cn2ccc3ccccc32)cc1", "ClCc1ccccc1.c1ccc2[nH]ccc2c1")

# Predict a file
MAX_LINES = 250
# Correct translations from from top-1 to top-5
cor_trans = [0, 0, 0, 0, 0]


def calc_accuracies(translations, ground_truth):
    global cor_trans
    for trans_index in range(len(translations)):
        if translations[trans_index] == ground_truth:
            # trans_index = 0, 4
            for acc_index in range(len(cor_trans) - trans_index):
                cor_trans[len(cor_trans) - acc_index - 1] += 1
            break


def print_accuracies(lines_count):
    global cor_trans
    for index, trans in enumerate(cor_trans):
        print('> Top ' + str(index + 1) + ': ' + str(trans / lines_count))


with io.open('data/retrosynthesis-test.smi') as data:
    lines = data.read().strip().split('\n')
    for i, line in enumerate(lines):
        print('\n> Iteration: ' + str(i + 1))
        if i >= MAX_LINES:
            break
        start = time.time()
        line = line.split(' >> ')
        translated, _, all_tokens = predict_smiles(smiles=line[0], expected=line[1])
        print(f'Time taken for 1 prediction: {time.time() - start:.2f} secs')
        calc_accuracies(all_tokens, ground_truth=line[1])
        print('> Current correct translations: ', cor_trans)
        print_accuracies(i+1)
