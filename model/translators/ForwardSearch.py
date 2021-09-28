from rdkit import Chem

from . import BeamSearchTranslator
from ..Transformer import Transformer


class ForwardSearchTranslator:

    def __init__(self, model: Transformer, forward_model: Transformer):
        self.model = model
        self.forward_model = forward_model

    def predict(self, sequence, tk, beam_size=5, validity_check=True, max_false_predictions=10, forward_beam_size=3):
        beam_search = BeamSearchTranslator(self.model)
        print('> Starting predictions ...')
        text, best_token, all_tokens = beam_search.predict(sequence, tk, beam_size=beam_size,
                                                           validity_check=validity_check,
                                                           max_false_predictions=max_false_predictions,
                                                           print_console=False)
        forward_beam = BeamSearchTranslator(self.forward_model)
        print("> Original model predictions: \t{}".format(all_tokens))

        # An array to save all predictions confirmed by the forward model by ranking
        predictions = []
        # An array to save which tokens of the original model have been taken
        taken_preds = []

        print('> Starting forward validation ...')
        # Check for each prediction whether the forward model would predict the original sequence
        for pred_index, token in enumerate(all_tokens):
            print('> Forward validation ' + str(pred_index + 1) + ' ...')
            _, _, forward_tokens = forward_beam.predict(token, tk, beam_size=forward_beam_size, print_console=False)

            for forward_index, forward_token in enumerate(forward_tokens):
                try:
                    # Check if the smiles strings are the same
                    if Chem.CanonSmiles(sequence) == Chem.CanonSmiles(forward_token):
                        # Create a ranking for the prediction
                        ranking = (pred_index / 2) + forward_index
                        # Save the prediction with its ranking
                        predictions.append([ranking, token])
                        # Save the index of the sequence
                        taken_preds.append(pred_index)
                        break
                except:
                    continue

        print('> Validated predictions: ' + str(len(predictions)))
        # Fill up the final result predictions of the model that have not been taken, if the beam size has not been
        # reached
        if len(predictions) < beam_size:
            for i in range(beam_size):
                if len(predictions) >= beam_size:
                    break
                if i not in taken_preds and len(all_tokens) > i:
                    predictions.append([i, all_tokens[i]])

        # Sort the predictions by their ranking
        predictions.sort(key=lambda x: x[0])
        # Create a final array containing only the sorted predictions validated by the forward model
        final = []
        for entry in predictions[:beam_size]:
            final.append(entry[1])

        print('> Prediction finished ... ')

        return final[0], best_token, final
