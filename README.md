# Retrosynthesis with Transformer

This project is part of the master thesis of Niklas Holtz with the topic "Retrosynthese mit Transformern" and contains all necessary scripts for the associated models, datasets and tools that were used in the thesis.
In the following a short overview should help to understand which script has what task and how the implemented tools can be used.

The implementation of the Transformer model, which can be found in the "model" folder, was done using the code published by Google. The code was therefore copied in places and is therefore not the result of this work. The adaptation of the Transformer to retrosynthesis, on the other hand, as well as the development of numerous other procedures, such as the beam search procedure, the forward reaction prediction, or even the complete automatic retrosynthesis have been developed exclusively for the underlying master thesis.

## Training and generating Output

The two most important scripts are `retro_prediction.py` and `retro_training.py` which are needed for training and applying the model.

For example, to train a model, the following command can be executed:

```
python retro_training.py --data_path=data/retrosynthesis-artificial.smi --batch_size=126 --epochs=200 --name=my_selfies_model --alphabet=alphabets/retrosynthesis-artificial-alphabet.txt
```

The `name` argument titles the name of the new model and `alphabet` specifies the path to an optional alphabet. Depending on whether this argument is set, the model will be trained using SELFIES notation. Note that hyperparamaters can also be specified as arguments. For more detailed information about the arguments, please refer to the corresponding script.
The finished model is saved in the directory along with a graphical representation for the accuracies and loss during training.

For example, to apply the model to a training data set, the following command can be executed:

```
python retro_prediction.py --model=trained_models/retro4 --test_data=data/retrosynthesis-test.smi --forward=trained_models/retro_forward
```

Here the `model` argument specifies the path to the trained model. The argument `test_data` specifies a file containing several chemical reactions. For these, the outputs are then iteratively generated and the accuracy of the model is calculated. Last but not least, the `forward` argument specifies the optional path to a model for forward reaction prediction.

However, it is also possible to get the output for a single input. This can be achieved for example with the following command.

```
python retro_prediction.py --model=trained_models/retro4 --smiles="Fc1cc2c(NC3CCCCCC3)ncnc2cn1"
```

This should generate the following output, where "Output smiles" is the output with the highest probability. However, the attached table also contains all other outputs calculated for the defined `beam_size`.

```
> Prediction finished ... 
Input smiles:   Fc1cc2c(NC3CCCCCC3)ncnc2cn1
Output smiles:  Fc1cc2c(Cl)ncnc2cn1.NC1CCCCCC1
Ground truth:   unknown
All predictions: ['Fc1cc2c(Cl)ncnc2cn1.NC1CCCCCC1', 'NC1CCCCCC1.Fc1cc2c(Cl)ncnc2cn1', 'Clc1ncnc2cnc(Cl)cc12.NC1CCCCCC1', 'Clc1ncnc2cnc(F)cc12.NC1CCCCCC1', 'O=c1[nH]cnc2cnc(F)cc12.NC1CCCCC1']
```

However, it is also possible to output an even simpler form that does not use the beam search procedure. If the argument "greedy" is set, the output with the highest probability is selected per input token. However, this does not lead to the best overall output, so Beam Search produces better results overall.

```
python retro_prediction.py --model=trained_models/retro4 --smiles="Fc1cc2c(NC3CCCCCC3)ncnc2cn1" --greedy=True
```

## Full Retrosynthesis

Within the scope of the master thesis, an approach was also developed to perform a fully automatic retrosynthesis. This can be executed for a given `product` and a selected `model` with the following command. The script uses an API key from Molport to check if a molecule is available online on their marketplace. However, the API key may no longer be valid, so a new one should be requested.
```
python full_retro/full_retro_analysis.py --model=trained_models/retro4 --product="Fc1cc2c(NC3CCCCCC3)ncnc2cn1"
```

## Additional Tools

This folder contains also some tools that are purely responsible for generating data. The names as well as a corresponding description can be taken from the following table

| Script | Description |
|:---:|:---:|
| alphabet_gen.py | Used to create an SELFIES alphabet out of a dataset. |
| artificial_data_gen.py | Generates a specified amount of artificial data for a dataset and stores it in a separate file. |
| combined_data_gen.py | A short script for creating a combined dataset from two files (used for the third dataset of the thesis). |
| forward_data_gen.py | Another short script for creating a forward dataset that can be used for forward reaction prediction. |
| text_mining_extraction.py | Extracts all chemical reactions from a text mining dataset and bundles them into a new file. The data from the text mining was too large for the repository, so it can be submitted upon request. |

