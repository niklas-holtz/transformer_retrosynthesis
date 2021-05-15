import numpy as np
import tensorflow as tf

from ..Transformer import Transformer


class SimpleTranslator():

    def __init__(self, model: Transformer):
        self.model = model

    def predict(self, sequence, tk, max_length=30):
        """
        :param sequence: the sequence to be translated
        :param tk: the tokenizer
        :param max_length: the maximum length of the search
        :return: the translated sequence
        """
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

        print('> Starting predictions ...')

        # Start Prediction
        for i in range(max_length):

            # Predict
            predictions, attention_weights = self.model(inp_sequence, output, False)
            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.argmax(predictions, axis=-1)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)  # tf.Tensor([[ 1 23]], shape=(1, 2), dtype=int64)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == eos_token:
                break

        tokens = output.numpy()[0]
        text = tk.detokenize(tokens)

        return text, tokens, attention_weights
