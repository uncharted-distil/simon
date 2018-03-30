# Simon
Character-level CNN+LSTM model for text classification.

The acronym Simon stands for Semantic Inference for the Modeling of ONtologies

This work is heavily influenced by the following academic article: http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classifica

# Getting Started

To get started, make sure you are using python v3.5+ and pip install via

`pip3 install git+https://github.com/NewKnowledge/simon`

Remember to first install all dependencies:

`pip3 install -r requirements.txt`

Then, study the scripts and pretrained models included in the Simon/scripts subdirectory.

The first included script is main_train_on_faker_data.py. This script will attempt to use CNN + LSTM to classify fake generated data into semantic types (ints, floats, strings, etc).  The app will generate x columns (data_cols) with y rows (data_count) in training mode, using the Faker library as the data source.  The code will automatically insert "null"s across the dataset, with a user-specified percentage (--nullpct flag). It will crop off each cell to have a max length of 20 chars, and will only look at the first 500 rows in a column. The training classes are specified by Categories.txt and types.json. The following command will achieve ~98% training/validation accuracy.

`python3 main_train_on_faker_data.py --data_count 500 --data_cols 1000 --nb_epoch 20 --nullpct .1 --batch_size 32`

A pretrained model is also included if one wants to proceed to performing semantic classification in predict mode (--no_train flag). Use the following command to generate some fake data and classify it using the included pretrained model (be sure to set checkpoint_dir to pretrained_models/ in script first, if not already set to it).

`python3 main_train_on_faker_data.py --data_count 1000 --data_cols 10 --nullpct .1 --no_train --config text-class.20-0.38.pkl`

The following command will evaluate the same model on a prespecified dataset and print a comparison of hand-labeled to predicted classes.

`python3 main_evaluate_model_on_dataset.py --config text-class.20-0.38.pkl`

Upcoming developments include transfer learning and both GPU and heterogeneous parallelization capabilities.

To run tensorflow's tensorboard, open up another terminal and run:

`python -m tensorflow.tensorboard --logdir=logs`

Then visit localhost:6006

# Noteworthy files
- main_faker_data.py - Main entry point for data type classification
- types.json - maps Faker method generators to fairly specific types.
- DataGenerator - creates the fake data
- Encoder - encodes/reverse-encodes input characters and data labels

# Other files
- FakeDataDescriptor - helper class to get a view of what Faker generates and how our types.json classifies it as

