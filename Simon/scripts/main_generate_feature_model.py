import csv
import time
import os.path
import numpy as np
import pandas as pd
from Simon import Simon
import json
import nltk

def main(datapath, email_index, execution_config, DEBUG):

    # set important parameters
    maxlen = 20
    max_cells = 500
    checkpoint_dir = "pretrained_models/"
    with open(checkpoint_dir + 'Categories_base.txt','r') as f:
        Categories = f.read().splitlines()
    category_count = len(Categories)

    # load specified execution configuration
    if execution_config is None:
        raise TypeError
    Classifier = Simon(encoder={}) # dummy text classifier
    config = Classifier.load_config(execution_config, checkpoint_dir)
    encoder = config['encoder']
    intermediate_model = Classifier.generate_feature_model(maxlen, max_cells, category_count, checkpoint_dir, config, DEBUG = DEBUG)

    # load sample email
    with open(datapath) as data_file:
        emails = data_file.readlines()
    sample_email = json.loads(emails[int(email_index)])['body']
    if DEBUG:
        print('DEBUG::sample email:')
        print(sample_email)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sample_email_sentence = tokenizer.tokenize(sample_email)
    sample_email_sentence = [elem[-maxlen:] for elem in sample_email_sentence] # truncate
    all_email_df = pd.DataFrame(sample_email_sentence,columns=['Email 0'])
    if DEBUG:
        print('DEBUG::the final shape is:')
        print(all_email_df.shape)
    all_email_df = all_email_df.astype(str)
    raw_data = np.asarray(all_email_df.ix[:max_cells-1,:]) #truncate to max_cells
    raw_data = np.char.lower(np.transpose(raw_data).astype('U'))

    # encode data 
    X = encoder.x_encode(raw_data, maxlen)

    # generate features for email
    y = intermediate_model.predict(X)
    # discard empty column edge case
    y[np.all(all_email_df.isnull(),axis=0)] = 0

    # print and return result
    print('\n128-d Simon Feature Vector:\n')
    print(y[0])
    return y[0]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='attempts to discern data types looking at columns holistically.')

    parser.add_argument('--datapath', dest='datapath',
                        help='datapath containing data (i.e. emails) for which to generate features')

    parser.add_argument('--index', dest='data_index',
                        help='index of data (i.e. single email) from datapath for which to generate features')

    parser.add_argument('--config', dest='execution_config',
                        help='execution configuration to load. contains max_cells, and encoder config.')

    parser.add_argument('--debug', dest='debug_config',default="True",
                        help='whether or not to print debug information.')

    args = parser.parse_args()

    main(args.datapath, args.data_index, args.execution_config,args.debug_config)
