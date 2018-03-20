# simon
Character-level CNN+LSTM model for text classification.

This work is heavily influenced by the following academic article: http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classifica

This app will attempt to use CNN + LSTM to classify fake generated data.  The app will generate x columns (data_cols) with y rows (data_count) each using Faker as the data source.  The code will automatically insert "null"s across the dataset. It will crop off each cell to have a max length of 20 chars, and will only look at the first 500 rows in a column.

This works best and tested in Python 3

To install all dependencies:
pip install -r requirements.txt

Noteworthy files:
main_faker_data.py - Main entry point for data type classification
main_transfer_to_datalake.py - trains on data in the datalake using the faker weights as initial condition for training ("transfer learning")

Other main_* files have self-documenting names. This will shortly be carefully documented (code will also need to be shortly re-structured to avoid code replication among the various main_*.py files). Importantly, now we can transfer to new categories, can handle categorical/ordinal variables and are able to do multiple-GPU (in-graph replication) training. Hyper-parameters have been tuned to make everything significantly faster as well (even in the single-GPU case). Additionally, we can handle various geographic variables (although kinks in this are still being worked out).

Types.json - maps Faker method generators to fairly specific types.
DataGenerator - creates the fake data
Encoder - encodes

Other files:
FakeDataDescriptor - helper class to get a view of what Faker generates and how our types.json classifies it as.


Current limitations:
   - probably highly overfit to Faker data
   - The model may benefit from analyzing each cell before aggregating across rows.
     
Usages:
# transfer learning
--data_count 1000 --data_cols 10 --no_train --config text-class.14-0.13.pkl

#basic and fast faker training - yields 50-70% accuracy

`python main_faker_data.py --data_count 100 --data_cols 100 --nb_epoch 20 --nullpct .1 --batch_size 10`


#98+% accuracy faker training:

`python main_faker_data.py --data_count 1000 --data_cols 10000 --nb_epoch 40 --nullpct .1 --batch_size 10`


#nearly 100% accuracy

`python main_faker_data.py --data_count 1000 --data_cols 10 --nullpct .1 --no_train --config text-class.36-0.02.pkl`

#nearly 0% accuracy

`python main_faker_data.py --data_count 10 --data_cols 10 --nullpct .1 --no_train --config text-class.36-0.02.pkl`

#to run tensorflow's tensorboard, open up another terminal and run:

`python -m tensorflow.tensorboard --logdir=logs`
#Then visit localhost:6006
