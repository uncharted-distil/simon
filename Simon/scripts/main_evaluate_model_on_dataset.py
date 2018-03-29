import csv
import time
import os.path
import numpy as np
from Simon import *
from Simon.Encoder import *
from Simon.DataGenerator import *
from Simon.LengthStandardizer import *

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

    # read unit test data
    dataset_name = "o_38" # o_38 or o_185
    if(DEBUG):
        print("DEBUG::BEGINNING UNIT TEST...")
    raw_data = pd.read_csv('unit_test_data/'+dataset_name+'.csv',dtype='str')

    with open('unit_test_data/'+dataset_name+'_Header.csv','r',newline='\n') as myfile:
        reader = csv.reader(myfile, delimiter=',')
        header = []
        for row in reader:
            header.append(row)
    
    variable_names = list(raw_data)
    
    nrows,ncols = raw_data.shape
    
    out = DataLengthStandardizerRaw(raw_data,max_cells)
    
    nx,ny = out.shape
    if(DEBUG):
        print("DEBUG::%d rows by %d columns in dataset"%(nx,ny))
        print("DEBUG::manually annotated header:")
        print(header)
        print("DEBUG::%d samples in header..."%len(header))
    
    tmp = np.char.lower(np.transpose(out).astype('U')) # transpose the data

    # build classifier model    
    Classifier = Simon(encoder=encoder) # text classifier for unit test
        
    model = Classifier.generate_model(maxlen, max_cells, category_count)

    Classifier.load_weights(checkpoint, None, model, checkpoint_dir)
    
    model_compile = lambda m: m.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['binary_accuracy'])
    
    model_compile(model)
    
    # encode the data and evaluate model
    X, y = encoder.encode_data(tmp, header, maxlen)
    if(DEBUG):
        print("DEBUG::y is:")
        print(y)
        print("DEBUG::The encoded labels (first row) are:")
        print(y[0,:])

    max_cells = encoder.cur_max_cells
    data = type('data_type', (object,), {'X_test': X, 'y_test':y})
    
    result = Classifier.evaluate_model(max_cells, model, data, encoder, p_threshold)
    
    
    # generate log
    f = open('evaluate_model_on_dataset.report','w')
    f.write('{:>25} | {:>40} | {:>30} | {:>30}'.format('VARIABLE NAME', 
        'MANUAL ANNOTATION', 'SIMON ANNOTATION', 'PROBABILITIES')+'\n')
    f.write(' \n')
    print('{:>25} | {:>40} | {:>30} | {:>30}'.format('VARIABLE NAME', 
        'MANUAL ANNOTATION', 'SIMON ANNOTATION', 'PROBABILITIES'))
    print(' ')
    for i in np.arange(out.shape[1]):
        print('{:>30} | {:>30} | {:>30} | {:>30}'.format(variable_names[i], 
        str(header[i]), str(result[0][i]), str(result[1][i])))
        f.write('{:>25} | {:>40} | {:>30} | {:>30}'.format(variable_names[i], 
        str(header[i]), str(result[0][i]), str(result[1][i]))+'\n')
    f.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='attempts to discern data types looking at columns holistically.')

    parser.add_argument('--config', dest='execution_config',
                        help='execution configuration to load. contains max_cells, and encoder config.')

    parser.add_argument('--debug', dest='debug_config',default="True",
                        help='whether or not to print debug information.')

    args = parser.parse_args()

    main(args.checkpoint,args.debug_config)
