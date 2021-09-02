import argparse
import io
import time

import numpy as np
import tensorflow as tf
import model as trans
from rdkit import Chem


def predict_smiles(smiles, translator, tk, expected="unknown"):
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


def calc_accuracies(translations, ground_truth, cor_trans):
    """
    Checks if there ground truth is included somewhere in the top-k predictions and counts to correct translations
    inside of an array.
    :param translations: an array of all predicted translations
    :param ground_truth: the actual ground truth for a given molecule
    :param cor_trans: an array to count all correct translations
    :return: None
    """
    ground_truth = Chem.CanonSmiles(ground_truth)

    for trans_index in range(len(translations)):
        trans = translations[trans_index]
        try:
            trans = Chem.CanonSmiles(trans)
        except:
            print("Apparently a non-valid smiles string was found: " + translations[trans_index])
            continue

        if trans == ground_truth:
            for acc_index in range(len(cor_trans) - trans_index):
                cor_trans[len(cor_trans) - acc_index - 1] += 1
            break


def print_accuracies(lines_count, cor_trans):
    """
    This method is used to output the top-k accuracies in the console.
    :param lines_count: The amount of lines that have been evaluated so far
    :param cor_trans: an array to count all correct translations
    :return: None
    """
    for index, trans in enumerate(cor_trans):
        print('> Top ' + str(index + 1) + ': ' + str(trans / lines_count))


def main():
    parser = argparse.ArgumentParser(description='Retro-Transformer')
    # Meta data
    parser.add_argument('--layers', type=int, default=4, help='Number of layer of the model.')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model.')
    parser.add_argument('--dff', type=int, default=512)
    parser.add_argument('--heads', type=int, default=8, help='Number of heads during the attention calculation.')
    parser.add_argument('--dropout', type=float, default=0.1, help='The dropout rate of the model while training.')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of each batch while training the model.')
    parser.add_argument('--warmup', type=int, default=4000, help='The warmup value of the learning rate optimizer.')
    # The required model for the prediction process
    parser.add_argument('--model', type=str, required=True,
                        help='The path of the model that is used for the prediction.')
    # For evaluating and predicting multiple lines and calculating the top-k accuracies
    parser.add_argument('--eval', type=bool, default=False, help='If true, the algorithm expects a test data set, '
                                                                 'which is read out line by line and for which the '
                                                                 'top-K accuracies are then calculated.')
    parser.add_argument('--test_data', type=str, default='', help='The path to a test data set, e.g. '
                                                                     'path/to/data.smi')
    parser.add_argument('--max_lines', type=int, default=6000, help='Defines a maximum amount of lines that should be '
                                                                    'evaluated.')
    # For predicting just a single line
    parser.add_argument('--smiles', type=str, default='', help='If the parameter "eval" is false, a single '
                                                               'translation is output for this smiles string.')

    args = parser.parse_args()

    # Tokenizer
    tk = trans.SmilesTokenizer()

    # Create the model
    transformer = trans.Transformer(
        num_layers=args.layers,
        d_model=args.d_model,
        num_heads=args.heads,
        dff=args.dff,
        input_vocab_size=tk.get_vocab_size(),
        target_vocab_size=tk.get_vocab_size(),
        pe_input=1000,
        pe_target=1000,
        rate=args.dropout)

    # Initialize the model by using an example
    example = 'COCCNc1c(C)cccc1C >> COCCO.Cc1cccc(C)c1N'
    line = example.split(' >> ')
    line[0] = tk.tokenize(line[0])
    line[1] = tk.tokenize(line[1])
    line = tf.keras.preprocessing.sequence.pad_sequences(line, value=0, padding='post', dtype='int64', maxlen=199)
    inp, tar = np.split(line, 2)
    tar_inp = tar[:, :-1]
    predictions, _ = transformer(inp, tar_inp, False)

    # Load saved trained_models
    transformer.load_weights(args.model)

    # Create the translator
    translator = trans.BeamSearchTranslator(transformer)

    if not args.eval:
        predict_smiles(args.smiles, translator=translator, tk=tk)
    else:
        # Correct translations from from top-1 to top-5
        cor_trans = [0, 0, 0, 0, 0]

        # Opens the test file and reads in all available data and starts predicting
        with io.open(args.test_data) as data:
            lines = data.read().strip().split('\n')
            # Iterate over each line and try to predicit
            for i, line in enumerate(lines):
                print('\n> Iteration: ' + str(i + 1))
                # Stop if a max_lines argument is set
                if i >= args.max_lines:
                    break

                # Predict and calculate the correct translations
                start = time.time()
                line = line.split(' >> ')
                translated, _, all_tokens = predict_smiles(smiles=line[0], expected=line[1], translator=translator, tk=tk)
                print(f'Time taken for 1 prediction: {time.time() - start:.2f} secs')
                calc_accuracies(all_tokens, ground_truth=line[1], cor_trans=cor_trans)
                print('> Current correct translations: ', cor_trans)
                print_accuracies(i + 1, cor_trans=cor_trans)


if __name__ == '__main__':
    main()
