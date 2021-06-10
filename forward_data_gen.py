import io

num_entries = None
path = "data/retrosynthesis-train.smi"

# Load the lines from the file and separate them
lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
print('Creating dataset for ' + str(num_entries) + ' out of ' + str(len(lines)) + ' found entries of the document.')
# Split the entries
word_pairs = [[w for w in l.split(' >> ')[0:2]] for l in lines[:num_entries]]


with open("data/retrosynthesis-train-forward.smi", 'w') as file:
    for i, pair in enumerate(word_pairs):
        prod = pair[0]
        reactant = pair[1]
        file.write(reactant + " >> " + prod + "\n")
