import argparse
import time
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model as trans

# To show proper values when using numpy
np.set_printoptions(precision=3)
np.set_printoptions(suppress=False)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def main():
    parser = argparse.ArgumentParser(description='Retro-Transformer')
    # Meta data
    parser.add_argument('--layers', type=int, default=4, help='Number of layer of the model.')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model.')
    parser.add_argument('--dff', type=int, default=512)
    parser.add_argument('--heads', type=int, default=8, help='Number of heads during the attention calculation.')
    parser.add_argument('--dropout', type=float, default=0.1, help='The dropout rate of the model while training.')
    parser.add_argument('--warmup', type=int, default=4000, help='The warmup value of the learning rate optimizer.')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of each batch during the training.')
    # Name and output path
    parser.add_argument('--name', type=str, default='retro_trans', help='The name of the model to be trained (without '
                                                                        '.h5 extension).')
    parser.add_argument('--output_path', type=str, default='', help='The output directory of the model. If no path is '
                                                                    'specified, the model is simply saved in the '
                                                                    'original directory.')
    # Optional plotting
    parser.add_argument('--plot', type=bool, default=True, help='Whether the plot of the loss and accuracy data of '
                                                                'the training should be plotted.')
    parser.add_argument('--csv', type=bool, default=True, help='Whether the plot data should be saved as a csv file.')
    # Required arguments
    parser.add_argument('--data_path', type=str, required=True, help='The path of the dataset that is used to train '
                                                                     'the model.')
    parser.add_argument('--epochs', type=int, default=100, required=True, help='Number of epochs the model is to be '
                                                                               'trained.')

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

    # Use a DatasetGenerator in order to load all data from a given path and combine it in a single dataset object
    generator = trans.DatasetGenerator(tk)
    count = None  # None means all
    dataset = generator.gen_from_file(args.data_path, count, args.batch_size, 0)

    # Schedule for the learning rate
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=args.warmup):
            super(CustomSchedule, self).__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    learning_rate = CustomSchedule(args.d_model)

    # Initialize an Adam Optimizer in order to improve the learning rate during the training
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    # Create a LossObject by using crossentropy
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    # Define the loss function for the training
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    # Define the accuracy function for the training
    def accuracy_function(real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    # A @tf.function improves the performance of the training and provides a faster execution.
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
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

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    # The main training method that uses the train_step method for each batch
    def main_train(dataset, n_epochs=args.epochs, print_every=50):
        losses = []
        accuracies = []

        for epoch in range(n_epochs):
            print("\nStarting epoch {}".format(epoch + 1))
            start = time.time()

            # Reset the loss and accuracy calculations
            train_loss.reset_states()
            train_accuracy.reset_states()

            # Get a batch of inputs and targets
            for (batch, (inp, tar)) in enumerate(dataset):
                train_step(inp, tar)

                # Print for each desired batch
                if batch % print_every == 0:
                    print("Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(epoch + 1, batch, train_loss.result(),
                                                                                 train_accuracy.result()))
            # Save losses and accuracies
            losses.append(train_loss.result())
            accuracies.append(train_accuracy.result())
            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
        return losses, accuracies

    # Start training
    losses, accuracies = main_train(dataset, print_every=100)

    # Adjust the output string if necessary
    directory = args.output_path
    if len(directory) > 0 and not directory.endswith('/'):
        directory += '/'

    # Save the model
    transformer.save_weights(directory + args.name + ".h5", save_format="h5")

    if args.plot:
        # Show some results from training
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        # plot some data
        ax1.plot(losses, label='loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        # accuracies
        ax2.plot(accuracies, label='acc')
        ax2.set_title('Training Accuracy')
        plt.savefig(directory + args.name + '_plot.png')
    if args.csv:
        # save plot data as csv
        with open(directory + args.name + '_plot_data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Losses', 'Accuracies'])
            for loss, acc in zip(losses, accuracies):
                writer.writerow([loss.numpy(), acc.numpy()])


if __name__ == '__main__':
    main()
