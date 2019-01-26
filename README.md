# Simon
Character-level CNN+LSTM model for text classification.

The acronym Simon stands for Semantic Inference for the Modeling of ONtologies.

Please see our paper on arxiv for more details about the implementation and some experimental results - https://arxiv.org/abs/1901.08456

You can also see NK-TexasAISummit.pdf at the top level of the repo for a brief introduction via our 2019 Texas AI Summit talk. 

This work is heavily influenced by the following academic article: http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classifica

# Getting Started

To get started, make sure you are using python v3.5+ and pip install via

`pip3 install git+https://github.com/NewKnowledge/simon`

Remember to first install all dependencies:

`pip3 install -r requirements.txt`

Then, study the scripts and pretrained models included in the Simon/scripts subdirectory.

The first included script is main_train_on_faker_data.py. This script will attempt to use CNN + LSTM to classify fake generated data into semantic types (ints, floats, strings, etc).  The app will generate x columns (data_cols) with y rows (data_count) in training mode, using the Faker library as the data source.  The code will automatically insert "null"s across the dataset, with a user-specified percentage (--nullpct flag). It will crop off each cell to have a max length of 20 chars, and will only look at the first 500 rows in a column. The training classes are specified by Categories.txt and types.json. The following command will achieve ~98% training/validation accuracy.

`python3 main_train_on_faker_data.py --data_count 500 --data_cols 1000 --nb_epoch 20 --nullpct .1 --batch_size 32`

To run the multigpu version of the same code, see main_train_on_faker_data.py script, and be sure to specify the number of GPUs available correctly. Also, you should be very judicious here about the cuda version you have installed, and whether it matches with a compatible version of installed tensorflow-gpu. A tensorflow-gpu version before 1.4.1 requires cuda 8.0, while later versions require version 9.0. The corresponding command is as follows.

`python3 main_train_on_faker_data_multi_gpu.py --data_count 500 --data_cols 1000 --nb_epoch 20 --nullpct .1 --batch_size 32`

A pretrained model is also included if one wants to proceed to performing semantic classification in predict mode (--no_train flag). Use the following command to generate some fake data and classify it using the included pretrained model (be sure to set checkpoint_dir to pretrained_models/ in script first, if not already set to it). This will classify into one of the base categories.

`python3 main_train_on_faker_data.py --data_count 1000 --data_cols 10 --nullpct .1 --no_train --config Base.pkl`

The following command will rigorously evaluate the same model on a prespecified dataset and print a comparison of hand-labeled to predicted classes. Please note that for this to work, the classes specified in the hand-labeled header cannot be outside those the specified model was trained to recognize, otherwise the software will error out.

`python3 main_evaluate_model_on_dataset.py --config Base.pkl`

To simply classify the columns of a dataset, execute the following command. The corresponding script also demonstrates how to use included rule-based classification functions for categorical/ordinal variables - however note that some of the included pretrained models can classify these using the neural net itself, but at a slightly lower accuracy.

`python3 main_classify_dataset.py --config Base.pkl`

Note that the code should work directly on a GPU-enabled machine. Be sure to set batch_size to something relatively large, like 64 (default is 5) to take full advantage of the GPU power with the flag addition `--batch_size 64`

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

