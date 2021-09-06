import argparse
import io
import time

import model as trans
from rdkit import Chem
from keras.models import load_model


def predict_reactants(smiles, translator, tk, expected="unknown"):
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
    # The required model for the prediction process
    parser.add_argument('--model', type=str, required=True,
                        help='The path of the model that is used for the prediction.')
    # For evaluating and predicting multiple lines and calculating the top-k accuracies
    parser.add_argument('--test_data', type=str, default='', help='The path to a test data set, e.g. '
                                                                     'path/to/data.smi')
    parser.add_argument('--max_lines', type=int, default=6000, help='Defines a maximum amount of lines that should be '
                                                                    'evaluated.')
    # For predicting just a single line
    parser.add_argument('--smiles', type=str, default='', help='If the parameter "eval" is false, a single '
                                                               'translation is output for this smiles string.')
    # Selfies
    parser.add_argument('--selfies', type=bool, default=False, help='If true, the model uses SELFIES instead of SMILES.')
    parser.add_argument('--alphabet', type=str, default='The alphabet that was used to train the model.')
    args = parser.parse_args()

    # Tokenizer
    if not args.selfies:
        tk = trans.SmilesTokenizer()
    else:
        tk = trans.SelfiesTokenizer()
        tk.load_alphabet_from_file(args.alphabet)

    # Load saved trained_models
    transformer = load_model(args.model)

    # Create the translator
    translator = trans.BeamSearchTranslator(transformer)

    if len(args.smiles) > 0:
        predict_reactants(args.smiles, translator=translator, tk=tk)
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
                translated, _, all_tokens = predict_reactants(smiles=line[0], expected=line[1], translator=translator, tk=tk)
                print(f'Time taken for 1 prediction: {time.time() - start:.2f} secs')
                calc_accuracies(all_tokens, ground_truth=line[1], cor_trans=cor_trans)
                print('> Current correct translations: ', cor_trans)
                print_accuracies(i + 1, cor_trans=cor_trans)


if __name__ == '__main__':
    main()
