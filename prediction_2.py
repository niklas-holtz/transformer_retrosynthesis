"""
Dieser Ansatz baut auf das Transformer-Paper auf und versucht einen anderen Ansatz des Beam Searches zu
implementieren, um so eine schnellere Ausführung zu ermöglichen.
"""
import io
import math
import os
import time

import numpy as np
import tensorflow as tf
from rdkit import Chem

from model.tokenizers.SmilesTokenizer import SmilesTokenizer
from model.Transformer import Transformer

# use a CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
batch_size = 128
EPOCHS = 200

# Tokenizer
tk = SmilesTokenizer()

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
transformer.load_weights('trained_models/tr-200e_128b.h5')


class suppress_stderr(object):
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR)]
        self.save_fds = [os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 2)

        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def gen_beams(sequence, beam_size=5, max_predict=160):
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

    lines = []
    scores = []
    final_beams = []
    vocab_size = tk.get_vocab_size()

    for i in range(beam_size):
        lines.append("")
        scores.append(0.0)

    start = time.time()
    for step in range(max_predict):
        if step > 0:
            print(f'Time taken for 1 step: {time.time() - start:.5f} secs')
            start = time.time()
        print("step = ", step)
        if step == 0:

            p, _ = transformer(inp_sequence, output, False)
            p = p[:, -1:, :]
            softmax = tf.keras.layers.Softmax()
            p = softmax(p)
            p = p.numpy()[0][0]

            nr = np.zeros((vocab_size, 2))
            for i in range(vocab_size):
                nr[i, 0] = - math.log10(p[i])
                nr[i, 1] = i
        else:
            cb = len(lines)
            nr = np.zeros((cb * vocab_size, 2))
            for i in range(cb):
                c_line = lines[i]  # z.B: 'C0C123'

                # tokenize the line
                c_tokenize = np.zeros(shape=(len(c_line)), dtype=np.int64)
                for i, char in enumerate(c_line.strip()):
                    c_tokenize[i] = tk.char_to_ix[char]

                # tf.Tensor([[1]], shape=(1, 1 .. 2.. 3.. . 4 usw.), dtype=int64)

                output = tf.convert_to_tensor(c_tokenize, np.int64)  # EagerTensor(1, 1, 31)
                output = tf.expand_dims(output, 0)

                p, _ = transformer(inp_sequence, output, False)
                p = p[:, -1:, :]
                softmax = tf.keras.layers.Softmax()
                p = softmax(p)
                p = p.numpy()[0][0]

                for j in range(vocab_size):
                    nr[i * vocab_size + j, 0] = - math.log10(p[j]) + scores[i]
                    nr[i * vocab_size + j, 1] = i * 100 + j

        y = nr[nr[:, 0].argsort()]

        new_beams = []
        new_scores = []

        for i in range(beam_size):

            c = tk.ix_to_char[y[i, 1] % 100]
            beamno = int(y[i, 1]) // 100

            if c == '$':
                added = lines[beamno] + c
                if added != "$":
                    final_beams.append([lines[beamno] + c, y[i, 0]])
                beam_size -= 1
            else:
                new_beams.append(lines[beamno] + c)
                new_scores.append(y[i, 0])

        lines = new_beams
        scores = new_scores

        if len(lines) == 0: break;

    for i in range(len(final_beams)):
        final_beams[i][1] = final_beams[i][1] / len(final_beams[i][0])

    final_beams = list(sorted(final_beams, key=lambda x: x[1]))[:5]
    answer = []

    for k in range(5):
        reags = set(final_beams[k][0].split("."))
        sms = set()

        with suppress_stderr():
            for r in reags:
                r = r.replace("$", "")
                m = Chem.MolFromSmiles(r)
                if m is not None:
                    sms.add(Chem.MolToSmiles(m))
                # print(sms);
            if len(sms):
                answer.append([sorted(list(sms)), final_beams[k][1]])

    return answer


def predict_smiles(smiles, expected=""):
    """
    Creates a prediction based on a single smiles string.
    :param smiles: the input smiles string
    :param expected: the expected output (ground truth)
    :return: the predicted smiles string (output)
    """
    translated_smiles, translated_tokens, all_tokens = gen_beams(smiles)
    print("Input smiles: \t{}".format(smiles))
    print("Output smiles: \t{}".format(translated_smiles))
    print("Ground truth: \t{}".format(expected))
    print("All predictions: {}".format(all_tokens))
    return translated_smiles, translated_tokens, all_tokens


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
        if i >= MAX_LINES:
            break
        print('\n> Iteration: ' + str(i + 1))
        start = time.time()
        line = line.split(' >> ')
        translated, _, all_tokens = predict_smiles(smiles=line[0], expected=line[1])
        print(f'Time taken for 1 prediction: {time.time() - start:.2f} secs')
        calc_accuracies(all_tokens, ground_truth=line[1])
        print('> Current correct translations: ', cor_trans)
        print_accuracies(i + 1)
