import csv
import io

path = 'data/retrosynthesis-artificial-2.smi'

path_dir = 'trained_models/deprecated/retro2'
file = 'slurm.train.100333.out'
name = 'retro2'
line_to_search = 'Batch 700'


lines = io.open(path_dir + '/' + file, encoding='UTF-8').read().strip().split('\n')

with open(path_dir + '/' + name + '_plot_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for index, l in enumerate(lines):
        if line_to_search in l:
            accuracy = l.split(' Accuracy ')[1]
            loss = l.split(' Accuracy ')[0].split('Loss ')[1]
            epoch = l.split(' Batch ')[0].split('Epoch ')[1]
            writer.writerow([epoch, loss, accuracy])
                # for loss, acc in zip(losses, accuracies):
                #    writer.writerow([loss.numpy(), acc.numpy()])
            print(accuracy)
            print(loss)
            print(epoch)
            print(l)


