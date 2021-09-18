import argparse
import model as trans
import tensorflow as tf
import logging
# Hide warnings
tf.get_logger().setLevel(logging.ERROR)
import greedy_analyser as ga


def main():
    parser = argparse.ArgumentParser(description='Full Retrosynthetic Analysis')
    # Input
    parser.add_argument('--product', type=str, default='',
                        help='The product for which a retrosynthetic analysis is to be performed.')
    # Model
    parser.add_argument('--model', type=str, default='trained_models/retro2_200e',
                        help='The path of the model that is used for the prediction.')
    # Translator arguments
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--max_false_predictions', type=int, default=30)

    args = parser.parse_args()

    # Tokenizer
    tk = trans.SmilesTokenizer()

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
    print("Starting retrosynthetic analysis for molecule: " + args.product)
    analyser = ga.GreedyAnalyser(translator, transformer)
    solution = analyser.analyse(args.product, tk)
    print("Retrosynthetic analysis result: ")
    print(solution)


if __name__ == '__main__':
    main()
