import argparse
import io
import time

import model as trans
from rdkit import Chem


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
    parser.add_argument('--selfies', type=bool, default=False,
                        help='If true, the model uses SELFIES instead of SMILES.')
    parser.add_argument('--alphabet', type=str, default='The alphabet that was used to train the model.')
    # Forward
    parser.add_argument('--forward', type=str, default='The path to a forward model in order to use the forward '
                                                       'reaction prediction.')


    args = parser.parse_args()

    # Tokenizer
    if not args.selfies:
        tk = trans.SmilesTokenizer()
    else:
        tk = trans.SelfiesTokenizer()
        tk.load_alphabet_from_file(args.alphabet)

    # Create the model
    transformer = trans.Transformer(
        num_layers=4,
        d_model=128,
        num_heads=8,
        dff=512,
        input_vocab_size=tk.get_vocab_size(),
        target_vocab_size=tk.get_vocab_size(),
        pe_input=1000,
        pe_target=1000,
        rate=0.1)

    # Load saved trained_models
    # Unfortunately I had issues with the RAM usage when load a saved model.
    # Take a look at the following post by me at stackoverflow to learn more:
    # https://stackoverflow.com/questions/69160914/why-does-load-model-cause-ram-memory-problems-while-predicting
    # transformer = load_model(args.model, compile=False)
    transformer.load_weights(args.model + "/variables/variables")

    # Create the translator
    if len(args.forward) > 0:
        # Use forward search
        forward_model = trans.Transformer(
            num_layers=4,
            d_model=128,
            num_heads=8,
            dff=512,
            input_vocab_size=tk.get_vocab_size(),
            target_vocab_size=tk.get_vocab_size(),
            pe_input=1000,
            pe_target=1000,
            rate=0.1)
        forward_model.load_weights(args.forward + "/variables/variables")
        translator = trans.ForwardSearchTranslator(transformer, forward_model)
    else:
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
            print('Starting evaluation for model: ' + args.model)
            for i, line in enumerate(lines):
                print('\n> Iteration: ' + str(i + 1))
                # Stop if a max_lines argument is set
                if i >= args.max_lines:
                    break

                # Predict and calculate the correct translations
                start = time.time()
                line = line.split(' >> ')
                translated, _, all_tokens = predict_reactants(smiles=line[0], expected=line[1], translator=translator,
                                                              tk=tk)
                print(f'Time taken for 1 prediction: {time.time() - start:.2f} secs')
                calc_accuracies(all_tokens, ground_truth=line[1], cor_trans=cor_trans)
                print('> Current correct translations: ', cor_trans)
                print_accuracies(i + 1, cor_trans=cor_trans)


if __name__ == '__main__':
    main()
