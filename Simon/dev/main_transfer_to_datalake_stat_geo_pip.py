import csv
import time
import os.path
import numpy as np

import azure_utils.client as client
import graphutils.getConnection as gc
from FetchLabeledData import *

from Simon import *
from Simon.Encoder import *
from Simon.DataGenerator import *
from Simon.LengthStandardizer import *

from Simon.penny.guesser import guess
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GeonamesCountriesTxtFileReader(object):
 """
  A GeonamesCountriesTxtFileReader 
 """
 def __init__(self, input_file):
  """Return a GeonamesCountriesTxtFileReader object"""   
  self.input_file = input_file

 def get_header_row_postal(self):
  return [ 
   'country_code',
   'postal_code',
   'city',
   'state',
   'admin-code-1',
   'admin-name-2',
   'admin-code-2',
   'admin-name-3',
   'admin-code-3',
   'latitude',
   'longitude',
   'accuracy' ]
   
 def get_data_types_postal(self):
  return { 
   'country_code' : str,
   'postal_code' : str,
   'city' : str,
   'state' : str,
   'admin-code-1' : str,
   'admin-name-2' : str,
   'admin-code-2' : str,
   'admin-name-3' : str,
   'admin-code-3' : str,
   'latitude' : str,
   'longitude' : str,
   'accuracy' : str }
  
 def read_csv_postal(self):
  start = time.time()
  df = pd.read_csv(
   self.input_file,
   delim_whitespace=False,
   sep='\t',
   error_bad_lines=False,
   skiprows=0,
   encoding='utf-8',
   names=self.get_header_row_postal(),
   dtype=self.get_data_types_postal(),
   na_values=['none'],
   usecols=self.get_header_row_postal())
  
  end = time.time()
  logger.info('Read CSV File (path = {}, elapsed-time = {})'.format(self.input_file, (end - start)))
  
  return df

def main(checkpoint, data_count, data_cols, should_train, nb_epoch, null_pct, try_reuse_data, batch_size, execution_config):
    maxlen = 20
    max_cells = 500

    # set this boolean to True to print debug messages (also terminate on *any* exception)
    DEBUG = True

    checkpoint_dir = "checkpoints/"
    
    # basic preliminaries for database calls
    store_name = 'nktraining'
    adl = client.get_adl_client(store_name)
    files = adl.ls('training-data/CKAN')
    random.shuffle(files)
    cnxn = gc.getConnection()
    cursor = cnxn.cursor()
    
    # make required database calls
    out, out_array_header = FetchLabeledDataFromDatabase(max_cells, cursor, adl, False)
    cnxn.close()
    
    with open('Categories_base_stat_geo.txt','r') as f:
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
    header = out_array_header
    
    ## ADD GEODATA
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # read-in relevant data
    in_file_postal = 'data/geodata/allCountries_postal.txt'
    reader_postal = GeonamesCountriesTxtFileReader(in_file_postal)
    df_postal = reader_postal.read_csv_postal()
    
    # read-in country codes
    in_file_cc = 'data/geodata/country_codes_csv.csv'
    df_countries = pd.read_csv(in_file_cc,dtype=str)

    # now, build up geo-data training data set
    N_SAMPLES = 500 # number of samples for each additional category
    
    start = time.time()
    data = np.asarray(df_countries['Name'].sample(n=max_cells,replace='True'))[np.newaxis].T # draw first random column
    data_header = list([['country','text'],]) # start with the 'country' category
    for i in np.arange(1,N_SAMPLES): # draw remaining random columns
        #print(data)
        data = np.concatenate((data,np.asarray(df_countries['Name'].sample(n=max_cells,replace='True'))[np.newaxis].T),axis=1)
        data_header.append(list(['country','text']))
    # now, handle the remaining new geo categories
    NewCategories = list(['state','city','postal_code','latitude','longitude','country_code'])
    for category in NewCategories:
        for i in np.arange(N_SAMPLES):
            data = np.concatenate((data,np.asarray(df_postal[category].sample(n=max_cells,replace='True'))[np.newaxis].T),axis=1)
            if((category=='latitude') or (category=='longitude')):
                data_header.append(list([category,'float']))
            elif(category=='postal_code'):
                # label as 'int' where appropriate (one may also need to label as 'text' sometimes as well, but going with this for now)
                data_header.append(list([category,'int']))
            else :
                data_header.append(list([category,'text']))
    
    if(DEBUG):
        print("DEBUG::the shape of geo data is:")
        print(data.shape)
        print("DEBUG::the length of the corresponding geo data header is:")
        print(len(data_header))
        print("DEBUG::the time elapsed to build the geo data set is (sec):")
        print(time.time()-start)
        print("DEBUG::merging geo data with datalake data... ")
    
    raw_data = np.column_stack((raw_data,data))
    header.extend(data_header)
    
    if(DEBUG):
        print("DEBUG::done!!")
        print("DEBUG::the shape of final merged data is:")
        print(raw_data.shape)
        print("DEBUG::the length of final merged data header is:")
        print(len(header))
    
    ## FINISHED ADDING GEODATA
    
    ## LABEL COMBINED DATA AS CATEGORICAL/ORDINAL
    start_time_guess = time.time()
    guesses = []
    print("Beginning Guessing categorical/ordinal for geo+datalake data...")
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
    category_count = len(Categories) # need to reset this variable as it is used for stg. else above
    
    # build classifier model    
    Classifier = Simon(encoder=encoder) # text classifier for unit test    
    model = Classifier.generate_transfer_model(maxlen, max_cells, category_count-9, category_count, checkpoint, checkpoint_dir)
    
    model_compile = lambda m: m.compile(loss='binary_crossentropy',
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
    
    ## Build p_threshold per class
    # p_threshold = np.sum(data.y_train,axis=0)*1/(data.y_train.shape[0])
    p_threshold = 0.5 # you could also use a fixed scalar
    
    if(DEBUG):
        print("DEBUG::per class p_threshold is:")
        print(p_threshold)

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