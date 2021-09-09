import argparse
import model as trans
from keras.models import load_model


def main():
    parser = argparse.ArgumentParser(description='Full Retrosynthetic Analysis')
    # Input
    parser.add_argument('--product', type=str, default='', help='The product for which a retrosynthetic analysis is to be performed.')
    # Model
    parser.add_argument('--model', type=str, default='trained_models/retro1', help='The path of the model that is used for the prediction.')
    # Translator arguments
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--max_false_predictions', type=int, default=30)

    args = parser.parse_args()

    print(args.product)

    # Tokenizer
    tk = trans.SmilesTokenizer()

    # Load saved trained_models
    transformer = load_model(args.model, compile=False)

    # Create the translator
    translator = trans.BeamSearchTranslator(transformer)

    _, scores, all_tokens = translator.predict(args.product, tk, beam_size=args.beam_size, max_false_predictions=args.max_false_predictions)
    print("All predictions: {}".format(all_tokens))
    print("Scores: {}".format(scores))
    for token in all_tokens:
        print("Predicting for ... " + token)
        _, scores, all_tokens = translator.predict(token, tk, beam_size=args.beam_size, max_false_predictions=args.max_false_predictions)
        reactants = all_tokens.split('.')


        print("OUT: {}".format(all_tokens))


if __name__ == '__main__':
    main()