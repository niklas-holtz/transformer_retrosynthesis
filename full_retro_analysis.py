import argparse
import model as trans
import tensorflow as tf
import logging

# Hide warnings
tf.get_logger().setLevel(logging.ERROR)
import beam_analyser as ga


def main():
    parser = argparse.ArgumentParser(description='Full Retrosynthetic Analysis')
    # Input
    parser.add_argument('--product', type=str, default='',
                        help='The product for which a retrosynthetic analysis is to be performed.')
    # Model
    parser.add_argument('--model', type=str, default='trained_models/retro2',
                        help='The path of the model that is used for the prediction.')
    # Translator arguments
    parser.add_argument('--beam_size', type=int, default=5)
    # Dictionary path
    parser.add_argument('--dict', type=str, default='data/full-dataset-adapted-canon.smi')

    parser.add_argument('--alphabet', type=str, default='', help='The alphabet that was used to train the model.')

    parser.add_argument('--hard', type=bool, default=False)

    parser.add_argument('-max_iter', type=int, default=12)

    args = parser.parse_args()

    # Tokenizer
    if len(args.alphabet) < 1:
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

    # Load the model weights
    transformer.load_weights(args.model + "/variables/variables")

    # Create the translator
    translator = trans.BeamSearchTranslator(transformer)
    print("> Starting retrosynthetic analysis for molecule: " + args.product)
    analyser = ga.BeamAnalyser(translator, transformer, args.dict)
    solution = analyser.analyse(args.product, tk, beam_size=args.beam_size, hard=args.hard, max_iter=args.max_iter)
    print("> Retrosynthetic analysis result for smiles: " + args.product)
    print(solution)


if __name__ == '__main__':
    main()
