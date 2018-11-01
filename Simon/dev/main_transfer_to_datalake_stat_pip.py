import csv
import time
import os.path
import numpy as np

import azure_utils.client as client
import graphutils.getConnection as gc
from FetchLabeledData import *

from Simon.penny.guesser import guess

from Simon import *
from Simon.Encoder import *
from Simon.DataGenerator import *
from Simon.LengthStandardizer import *

def main(checkpoint, data_count, data_cols, should_train, nb_epoch, null_pct, try_reuse_data, batch_size, execution_config):
    maxlen = 20
    max_cells = 500
    p_threshold = 0.5

    # set this boolean to True to print debug messages (also terminate on *any* exception)
    DEBUG = False

    checkpoint_dir = "checkpoints/"
    
    # basic preliminaries for database calls
    store_name = 'nktraining'
    adl = client.get_adl_client(store_name)
    files = adl.ls('training-data/CKAN')
    random.shuffle(files)
    cnxn = gc.getConnection()
    cursor = cnxn.cursor()
    
    # make required database calls
    out, out_array_header = FetchLabeledDataFromDatabase(max_cells, cursor, adl, DEBUG)
    cnxn.close()
    
    with open('Categories_base_stat.txt','r') as f:
            Categories = f.read().splitlines()
    Categories = sorted(Categories)
    category_count = len(Categories)
    nx,ny = out.shape
    if(DEBUG):
        # orient the user a bit
        print("read data is: ")
        print(out)
        print("fixed categories are: ")
        print(Categories)
        print("The size of the read raw data is %d rows by %d columns"%(nx,ny))
        print("corresponding header length is: %d"%len(out_array_header))
        #print(out_array_header)
    
    # specify data for the transfer-learning experiment
    raw_data = out[0:max_cells,:] #truncate the rows (max_cells)
    
    #read "post-processed" header from file, this includes categorical and ordinal classifications...
#    with open('datalake_labels','r',newline='\n') as myfile:
#        reader = csv.reader(myfile, delimiter=',')
#        header = []
#        for row in reader:
#            header.append(row)
    # OR 
    header = out_array_header
    
    ## LABEL COMBINED DATA AS CATEGORICAL/ORDINAL
    start_time_guess = time.time()
    guesses = []
    print("Beginning Guessing categorical/ordinal for datalake data...")
    category_count = 0
    ordinal_count = 0
    for i in np.arange(raw_data.shape[1]):
        tmp = guess(raw_data[:,i], for_types ='category')
        if tmp[0]=='category':
            category_count += 1
            header[i].append('categorical')
            if ('int' in header[i]) or ('float' in header[i]) \
                or ('datetime' in header[i]):
                    ordinal_count += 1
                    header[i].append('ordinal')
        guesses.append(tmp)
    
    if(DEBUG):
        #print(guesses)
        #print(len(guesses))
        print("DEBUG::The number of categorical columns is %d"%category_count)
        print("DEBUG::The number of ordinal columns is %d"%ordinal_count)
        #print(header)
        
    elapsed_time = time.time()-start_time_guess
    print("Total guessing time is : %.2f sec" % elapsed_time)
    ## FINISHED LABELING COMBINED DATA AS CATEGORICAL/ORDINAL
    
    # transpose the data
    raw_data = np.char.lower(np.transpose(raw_data).astype('U'))
    
    # do other processing and encode the data
    if execution_config is None:
        raise TypeError
        
    Classifier = Simon(encoder={}) # dummy text classifier
    config = Classifier.load_config(execution_config, checkpoint_dir)
    encoder = config['encoder']
    checkpoint = config['checkpoint']
    
    encoder.categories=Categories
    
    # build classifier model    
    Classifier = Simon(encoder=encoder) # text classifier for unit test    
    model = Classifier.generate_transfer_model(maxlen, max_cells, category_count-2, category_count, checkpoint, checkpoint_dir)
    
    model_compile = lambda m: m.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['binary_accuracy'])
    model_compile(model)
    
    # encode the data and evaluate model
    X, y = encoder.encode_data(raw_data, header, maxlen)
    if(DEBUG):
        print("DEBUG::y is:")
        print(y)
        print("DEBUG::The encoded labels (first row) are:")
        print(y[0,:])
        
    data = Classifier.setup_test_sets(X, y)

    max_cells = encoder.cur_max_cells
    
    start = time.time()
    history = Classifier.train_model(batch_size, checkpoint_dir, model, nb_epoch, data)
    end = time.time()
    print("Time for training is %f sec"%(end-start))
    
    config = { 'encoder' :  encoder,
               'checkpoint' : Classifier.get_best_checkpoint(checkpoint_dir) }
    Classifier.save_config(config, checkpoint_dir)
    Classifier.plot_loss(history) #comment out on docker images...
    
    Classifier.evaluate_model(max_cells, model, data, encoder, p_threshold)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='attempts to discern data types looking at columns holistically.')

    parser.add_argument('--cp', dest='checkpoint',
                        help='checkpoint to load')

    parser.add_argument('--config', dest='execution_config',
                        help='execution configuration to load. contains max_cells, and encoder config.')

    parser.add_argument('--train', dest='should_train', action="store_true",
                        default="True", help='run training')
    parser.add_argument('--no_train', dest='should_train', action="store_false",
                        default="True", help='do not run training')
    parser.set_defaults(should_train=True)

    parser.add_argument('--data_count', dest='data_count', action="store", type=int,
                        default=100, help='number of data rows to create')
    parser.add_argument('--data_cols', dest='data_cols', action="store", type=int,
                        default=10, help='number of data cols to create')
    parser.add_argument('--nullpct', dest='null_pct', action="store", type=float,
                        default=0, help='percent of Nulls to put in each column')

    parser.add_argument('--nb_epoch', dest='nb_epoch', action="store", type=int,
                        default=5, help='number of epochs')
    parser.add_argument('--try_reuse_data', dest='try_reuse_data', action="store_true",
                        default="True", help='loads existing data if the dimensions have been stored')
    parser.add_argument('--force_new_data', dest='try_reuse_data', action="store_false",
                        default="True", help='forces the creation of new data, even if the dimensions have been stored')
    parser.add_argument('--batch_size', dest='batch_size', action="store", type=int,
                        default=5, help='batch size for training')

    args = parser.parse_args()

    main(args.checkpoint, args.data_count, args.data_cols, args.should_train,
         args.nb_epoch, args.null_pct, args.try_reuse_data, args.batch_size, args.execution_config)