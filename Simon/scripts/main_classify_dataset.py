import csv
import time
import os.path
import numpy as np
from Simon import *
from Simon.Encoder import *
from Simon.DataGenerator import *
from Simon.LengthStandardizer import *
from Simon.penny.guesser import guess

def main(execution_config, DEBUG):
    maxlen = 20
    max_cells = 500
    p_threshold = 0.5

    DEBUG = True # boolean to specify whether or not print DEBUG information

    checkpoint_dir = "pretrained_models/"

    with open('Categories.txt','r') as f:
        Categories = f.read().splitlines()
    
    # orient the user a bit
    print("fixed categories are: ")
    Categories = sorted(Categories)
    print(Categories)
    category_count = len(Categories)

    # load specified execution configuration
    if execution_config is None:
        raise TypeError
    Classifier = Simon(encoder={}) # dummy text classifier
    config = Classifier.load_config(execution_config, checkpoint_dir)
    encoder = config['encoder']
    checkpoint = config['checkpoint']

    # read unit test data
    dataset_name = "o_38" # o_38 or o_185
    if(DEBUG):
        print("DEBUG::BEGINNING UNIT TEST...")
    frame = pd.read_csv('unit_test_data/'+dataset_name+'.csv',dtype='str')

    X = encoder.encodeDataFrame(frame)
            
    # build classifier model
    model = Classifier.generate_model(maxlen, max_cells, category_count)
    Classifier.load_weights(checkpoint, None, model, checkpoint_dir)
    model_compile = lambda m: m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    model_compile(model)
    y = model.predict(X)
    # discard empty column edge case
    y[np.all(frame.isnull(),axis=0)]=0

    result = encoder.reverse_label_encode(y,p_threshold)

    ## LABEL COMBINED DATA AS CATEGORICAL/ORDINAL
    print("Beginning Guessing categorical/ordinal classifications...")
    start_time_guess = time.time()
    category_count = 0
    ordinal_count = 0
    raw_data = frame.as_matrix()
    for i in np.arange(raw_data.shape[1]):
        tmp = guess(raw_data[:,i], for_types ='category')
        if tmp[0]=='category':
            category_count += 1
            tmp2 = list(result[0][i])
            tmp2.append('categorical')
            result[0][i] = tuple(tmp2)
            result[1][i].append(1)
            if ('int' in result[1][i]) or ('float' in result[1][i]) \
                or ('datetime' in result[1][i]):
                    ordinal_count += 1
                    tmp2 = list(result[0][i])
                    tmp2.append('ordinal')
                    result[0][i] = tuple(tmp2)
                    result[1][i].append(1)
    elapsed_time = time.time()-start_time_guess
    print("Total statistical variable guessing time is : %.2f sec" % elapsed_time)
    ## FINISHED LABELING COMBINED DATA AS CATEGORICAL/ORDINAL
    print("The predicted classes and probabilities are respectively:")
    print(result)
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='attempts to discern data types looking at columns holistically.')

    parser.add_argument('--config', dest='execution_config',
                        help='execution configuration to load. contains max_cells, and encoder config.')

    parser.add_argument('--debug', dest='debug_config',default="True",
                        help='whether or not to print debug information.')

    args = parser.parse_args()

    main(args.execution_config,args.debug_config)
