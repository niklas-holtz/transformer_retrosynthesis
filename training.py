import io
import time

import numpy as np
import tensorflow as tf

# To show proper values when using numpy
#from model.tokenizers.SmilesTokenizer import SmilesTokenizer
#from model.Transformer import Transformer

import model as trans

np.set_printoptions(precision=3)
np.set_printoptions(suppress=False)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# Some hyper variables
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.25
batch_size = 32
EPOCHS = 20

# Tokenizer
tk = trans.tokenizers.SmilesTokenizer()

# Create the model
transformer = trans.Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tk.get_vocab_size(),
    target_vocab_size=tk.get_vocab_size(),
    pe_input=1000,
    pe_target=1000,
    rate=dropout_rate)


# Tokenize the data
def preprocess_entry(w):
    tokenized = tk.tokenize(w)
    # print("its ... " + w)
    # print("which is ... ", tokenized)
    return tokenized


# Load dataset
def load_data_from_file(path, num_entries):
    # Load the lines from the file and separate them
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    print('Creating dataset for ' + str(num_entries) + ' out of ' + str(len(lines)) + ' found entries of the document.')
    # Preprocess data, split the entries and create a zip object
    word_pairs = [[preprocess_entry(w) for w in l.split(' >> ')[0:2]] for l in lines[:num_entries]]
    return zip(*word_pairs)


def create_dataset(path, num_entries) -> tf.data.Dataset:
    # Load data from path
    products, reactants = load_data_from_file(path, num_entries)

    # Add padding
    combined = np.concatenate((products, reactants))
    combined = tf.keras.preprocessing.sequence.pad_sequences(combined, value=0, padding='post', dtype='int64')
    products, reactants = np.split(combined, 2)

    # print(products[-1])
    # print(reactants[-1])

    # Define a dataset object
    dataset = tf.data.Dataset.from_tensor_slices((products, reactants))
    dataset = dataset.shuffle(len(products), reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


path = 'data/retrosynthesis-train.smi'
count = None
dataset = create_dataset(path, count)


# print(dataset)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


def main_train(dataset, transformer, n_epochs=EPOCHS, print_every=50):
    losses = []
    accuracies = []

    for epoch in range(n_epochs):
        print("\nStarting epoch {}".format(epoch + 1))
        start = time.time()

        # Reset the losss and accuracy calculations
        train_loss.reset_states()
        train_accuracy.reset_states()

        # Get a batch of inputs and targets
        for (batch, (inp, tar)) in enumerate(dataset):
            # Set the decoder inputs
            tar_inp = tar[:, :-1]
            # Set the target outputs, right shifted
            tar_real = tar[:, 1:]

            # Create masks
            # enc_padding_mask, combined_mask, dec_padding_mask = transformer.create_masks(inp, tar_inp)

            with tf.GradientTape() as tape:
                # Call the transformer and get the predicted output
                predictions, _ = transformer(inp, tar_inp, True)
                # Calculate the loss
                loss = loss_function(tar_real, predictions)

            # Update the weights and optimizer
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            # Save and store the metrics
            train_loss(loss)
            train_accuracy(accuracy_function(tar_real, predictions))

            # Print and save losses and accuracies
            if batch % print_every == 0:
                losses.append(train_loss.result())
                accuracies.append(train_accuracy.result())
                print("Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(epoch + 1, batch, train_loss.result(),
                                                                             train_accuracy.result()))

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')  #
    return losses, accuracies

tf.debugging.set_log_device_placement(False)

# Start training
with tf.device('/GPU:0'):
    losses, accuracies = main_train(dataset, transformer, print_every=100)

# Model name
model_name = "tr-6"

# Save the model
transformer.save_weights(model_name + ".h5", save_format="h5")
