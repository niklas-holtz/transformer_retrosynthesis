import math

import numpy as np
import tensorflow as tf
from rdkit import Chem

from . import BeamSearchTranslator
from ..Transformer import Transformer


class ForwardSearchTranslator:

    def __init__(self, model: Transformer, forward_model: Transformer):
        self.model = model
        self.forward_model = forward_model

    def predict(self, sequence, tk, validity_check=True, max_false_predictions=10):
        beam_search = BeamSearchTranslator(self.model)
        text, best_token, all_tokens = beam_search.predict(sequence, tk, validity_check=validity_check, max_false_predictions=max_false_predictions)
        forward_beam = BeamSearchTranslator(self.forward_model)

        print("All predictions: \t{}".format(all_tokens))

        for token in all_tokens:
            _, _, forward_tokens = forward_beam.predict(token, tk, beam_size=3)

            print("Forward prediction of: \t{}".format(token))
            print("> Predictions: \t{}".format(forward_tokens))

        return text, best_token, all_tokens


