from DataGenerator import *
from DataLengthRandomizer import *
from DataLengthStandardizer import *
from Encoder import *
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Dropout, MaxPooling1D, Convolution1D
from keras.layers import LSTM, Lambda, merge, Masking
from keras.layers import Embedding, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
import re
from keras import backend as K
import keras.callbacks
import sys
import os
import time
import matplotlib.pyplot as plt
import pickle

import azure_utils.client as client
import graphutils.printSample as printSample
import graphutils.getConnection as gc
import graphutils.insertColumnDatasetJoin as insert

import pandas
import sys
import random
import pyodbc

import timeit
from LengthStandardizer import *
from FetchLabeledData import *


def binarize(x, sz=71):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))
    
def custom_multi_label_accuracy(y_true, y_pred):
    # need some threshold-specific rounding code here, presently only for 0.5 thresh.
    return K.mean(K.round(np.multiply(y_true,y_pred)),axis=0)
    
def eval_binary_accuracy(y_test, y_pred):
    correct_indices = y_test==y_pred
    all_correct_predictions = np.zeros(y_test.shape)
    all_correct_predictions[correct_indices] = 1
    #print("DEBUG::binary accuracy matrix")
    #print(all_correct_predictions)
    return np.mean(all_correct_predictions),np.mean(all_correct_predictions, axis=0),all_correct_predictions
    
def eval_confusion(y_test, y_pred):
    wrong_indices = y_test!=y_pred
    all_wrong_predictions = np.zeros(y_test.shape)
    all_wrong_predictions[wrong_indices] = 1
    #print("DEBUG::confusion matrix")
    #print(all_wrong_predictions)
    return np.mean(all_wrong_predictions),np.mean(all_wrong_predictions, axis=0),all_wrong_predictions
    
def eval_false_positives(y_test, y_pred):
    false_positive_matrix = np.zeros((y_test.shape[1],y_test.shape[1]))
    false_positives = np.multiply(y_pred,1-y_test)
    # print(precision_matrix)
    for i in np.arange(y_test.shape[0]):
        for j in np.arange(y_test.shape[1]) :
            if(false_positives[i,j]==1): #false positive label for ith sample and jth predicted category
                for k in np.arange(y_test.shape[1]):
                    if(y_test[i,k]==1): #positive label for ith sample and kth true category
                        # print("DEBUG::i,j,k")
                        # print("%d,%d,%d"%(i,j,k))
                        false_positive_matrix[j,k] +=1
    # print("DEBUG::precision matrix")
    # print(precision_matrix)
    return np.sum(false_positive_matrix),np.sum(false_positive_matrix, axis=0),false_positive_matrix
    


def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 71

def max_1d(x):
    return K.max(x, axis=1)


def striphtml(html):
    p = re.compile(r'<.*?>')
    return p.sub('', html)


def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)

def setup_test_sets(X, y):
    ids = np.arange(len(X))
    np.random.shuffle(ids)

    # shuffle
    X = X[ids]
    y = y[ids]

    train_end = int(X.shape[0] * .6)
    cross_validation_end = int(X.shape[0] * .3 + train_end)
    test_end = int(X.shape[0] * .1 + cross_validation_end)
    
    X_train = X[:train_end]
    X_cv_test = X[train_end:cross_validation_end]
    X_test = X[cross_validation_end:test_end]

    y_train = y[:train_end]
    y_cv_test = y[train_end:cross_validation_end]
    y_test = y[cross_validation_end:test_end]

    data = type('data_type', (object,), {'X_train' : X_train, 'X_cv_test': X_cv_test, 'X_test': X_test, 'y_train': y_train, 'y_cv_test': y_cv_test, 'y_test':y_test})
    return data


def generate_model(max_len, max_cells, category_count):
    filter_length = [1, 3, 3]
    nb_filter = [40, 200, 1000]
    pool_length = 2
    # document input
    document = Input(shape=(max_cells, max_len), dtype='int64')
    # sentence input
    in_sentence = Input(shape=(max_len,), dtype='int64')
    # char indices to one hot matrix, 1D sequence to 2D
    embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)
    # embedded: encodes sentence
    for i in range(len(nb_filter)):
        embedded = Convolution1D(nb_filter=nb_filter[i],
                                 filter_length=filter_length[i],
                                 border_mode='valid',
                                 activation='relu',
                                 init='glorot_normal',
                                 subsample_length=1)(embedded)

        embedded = Dropout(0.1)(embedded)
        embedded = MaxPooling1D(pool_length=pool_length)(embedded)

    forward_sent = LSTM(256, return_sequences=False, dropout_W=0.2,
                        dropout_U=0.2, consume_less='gpu')(embedded)
    backward_sent = LSTM(256, return_sequences=False, dropout_W=0.2,
                         dropout_U=0.2, consume_less='gpu', go_backwards=True)(embedded)

    sent_encode = merge([forward_sent, backward_sent],
                        mode='concat', concat_axis=-1)
    sent_encode = Dropout(0.3)(sent_encode)
    # sentence encoder

    encoder = Model(input=in_sentence, output=sent_encode)

    print(encoder.summary())
    encoded = TimeDistributed(encoder)(document)

    # encoded: sentences to bi-lstm for document encoding
    forwards = LSTM(128, return_sequences=False, dropout_W=0.2,
                    dropout_U=0.2, consume_less='gpu')(encoded)
    backwards = LSTM(128, return_sequences=False, dropout_W=0.2,
                     dropout_U=0.2, consume_less='gpu', go_backwards=True)(encoded)

    merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
    output = Dropout(0.3)(merged)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.3)(output)
    output = Dense(category_count, activation='sigmoid')(output)
    # output = Activation('softmax')(output)
    model = Model(input=document, output=output)
    return model


# record history of training
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('binary_accuracy'))
def plot_loss(history):
    # summarize history for accuracy
    plt.subplot('121')
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    plt.subplot('122')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def train_model(batch_size, checkpoint_dir, model, nb_epoch, data):
    print("starting learning")
    
    check_cb = keras.callbacks.ModelCheckpoint(checkpoint_dir + "text-class" + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                               monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0,
                                embeddings_layer_names=None, embeddings_metadata=None)
    loss_history = LossHistory()
    history = model.fit(data.X_train, data.y_train, validation_data=(data.X_cv_test, data.y_cv_test), batch_size=batch_size,
              nb_epoch=nb_epoch, shuffle=True, callbacks=[earlystop_cb, check_cb, loss_history, tbCallBack])
    
    print('losses: ')
    print(history.history['loss'])
    print('accuracies: ')
    #print(history.history['acc'])
    print(history.history['val_binary_accuracy'])
    plot_loss(history)

def evaluate_model(max_cells, model, data, encoder):
    print("Starting predictions:")
    
    start = time.time()
    scores = model.evaluate(data.X_test, data.y_test, verbose=0)
    end = time.time()
    print("Accuracy: %.2f%% \n Time: {0}s \n Time/example : {1}s/ex".format(
        end - start, (end - start) / data.X_test.shape[0]) % (scores[1] * 100))
    
    # return all predictions above a certain threshold
    # first, the maximum probability/class
    probabilities = model.predict(data.X_test, verbose=1)
    # print("The prediction probabilities are:")
    # print(probabilities)
    m = np.amax(probabilities, axis=1)
    max_index = np.argmax(probabilities, axis=1)
    # print("Associated fixed category indices:")
    # print(max_index)
    with open('Categories.txt','r') as f:
            Categories = f.read().splitlines()
    print("Remember that the fixed categories are:")
    print(Categories)
    print("Most Likely Predicted Category/Labels are: ")
    print((np.array(Categories))[max_index])
    print("Associated max probabilities/confidences:")
    print(m)
    # next, all probabilities above a certain threshold
    print("DEBUG::y_test:")
    print(data.y_test)
    p_threshold = 0.8
    prediction_indices = probabilities > p_threshold
    y_pred = np.zeros(data.y_test.shape)
    y_pred[prediction_indices] = 1
    print("DEBUG::y_pred:")
    print(y_pred)
    print("'Binary' accuracy (true positives + true negatives) is:")
    print(eval_binary_accuracy(data.y_test,y_pred))
    print("'Binary' confusion (false positives + false negatives) is:")
    print(eval_confusion(data.y_test,y_pred))
    print("False positive matrix is:")
    print(eval_false_positives(data.y_test,y_pred))
    
#    print("FINAL multilabel accuracy vector is:")
#    print(multi_label_accuracy(data.y_test, predictions))
    
#    match = []
#    mismatch = []
#    for val in results:
#        if val[1] == val[2][0][0]:
#            match.append(val)
#        else:
#            mismatch.append(val)
#    
#    # with open("test_results.txt", 'w') as results_file:
#    #    pprint.pprint(results, results_file)
#    with open("test_match_results.txt", 'w') as results_file:
#        pprint.pprint(match, results_file)
#    with open("test_mismatch_results.txt", 'w') as results_file:
#        pprint.pprint(mismatch, results_file)
#    
#    pprint.pprint(mismatch)

def main(checkpoint, data_count, data_cols, should_train, nb_epoch, null_pct, try_reuse_data, batch_size, execution_config):
    maxlen = 20
    max_cells = 500

    # set this boolean to True to print debug messages (also terminate on *any* exception)
    DEBUG = False

    checkpoint_dir = "checkpoints/"
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

        
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
    
    
    # orient the user a bit
    print("read clustered data is: ")
    print(out)
    print("fixed categories are: ")
    with open('Categories.txt','r') as f:
            Categories = f.read().splitlines()
    print(Categories)
    
    #label = input("Your best bet for the column label is? Select one of the fixed categories... : ")
    
    nx,ny = out.shape
    print("The size of the read raw data is %d rows by %d columns"%(nx,ny))
    print("corresponding header length is: %d"%len(out_array_header))
    #print(out_array_header)
    
       

    # specify data for the transfer-learning experiment
    raw_data = out[0:max_cells,:] #truncate the rows (max_cells)
    header = out_array_header
    
    
    
    # do other processing and encode the data
    if null_pct > 0:
        DataGenerator.add_nulls_uniform(raw_data, null_pct)
    config = {}
    if not should_train:
        if execution_config is None:
            raise TypeError
        config = load_config(execution_config, checkpoint_dir)
        encoder = config['encoder']
        if checkpoint is None:
            checkpoint = config['checkpoint']
    else:
        encoder = Encoder()
        encoder.process(raw_data, max_cells)
        
    # enable training mode for transfer-learning experiment
    should_train = True
    
    # transpose the data
    raw_data = np.char.lower(np.transpose(raw_data).astype('U'))
    
    # encode the data 
    X, y = encoder.encode_data(raw_data, header, maxlen)
    
    print("DEBUG::The encoded labels (one row) are:")
    print(y[0,:])

    max_cells = encoder.cur_max_cells
    data = None
    if should_train:
        data = setup_test_sets(X, y)
    else:
        data = type('data_type', (object,), {'X_test': X, 'y_test':y})

    print('Sample chars in X:{}'.format(X[2, 0:10]))
    print('y:{}'.format(y[2]))
    
    # need to know number of fixed categories to create model
    category_count = y.shape[1] 
    print('Number of fixed categories is :')
    print(category_count)

    model = generate_model(maxlen, max_cells, category_count)


    load_weights(checkpoint, config, model, checkpoint_dir)
    
    # retrain the last layer
    for layer in model.layers[:7]:
        layer.trainable = False
    model.layers[8].trainable = True


    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['binary_accuracy'])
    if(should_train):
        start = time.time()
        train_model(batch_size, checkpoint_dir, model, nb_epoch, data)
        end = time.time()
        print("Time for training is %f sec"%(end-start))
        config = { 'encoder' :  encoder,
                   'checkpoint' : get_best_checkpoint(checkpoint_dir) }
        save_config(config, checkpoint_dir)
        
    evaluate_model(max_cells, model, data, encoder)

def resolve_file_path(filename, dir):
    if os.path.isfile(str(filename)):
        return str(filename)
    elif os.path.isfile(str(dir + str(filename))):
        return dir + str(filename)

def load_weights(checkpoint_name, config, model, checkpoint_dir):
    if config and not checkpoint_name:
        checkpoint_name = config['checkpoint']
    if checkpoint_name:
        checkpoint_path = resolve_file_path(checkpoint_name, checkpoint_dir)
        print("Checkpoint : %s" % str(checkpoint_path))
        model.load_weights(checkpoint_path)

def save_config(execution_config, checkpoint_dir):
    filename = ""
    if not execution_config["checkpoint"] is None:
        filename = execution_config["checkpoint"].rsplit( ".", 1 )[ 0 ] + ".pkl"
    else:
        filename = time.strftime("%Y%m%d-%H%M%S") + ".pkl"
    with open(checkpoint_dir + filename, 'wb') as output:
        pickle.dump(execution_config, output, pickle.HIGHEST_PROTOCOL)

def load_config(execution_config_path, dir):
    execution_config_path = resolve_file_path(execution_config_path, dir)
    return pickle.load( open( execution_config_path, "rb" ) )

def get_best_checkpoint(checkpoint_dir):
    max_mtime = 0
    max_file = ''
    for dirname,subdirs,files in os.walk(checkpoint_dir):
        for fname in files:
            full_path = os.path.join(dirname, fname)
            mtime = os.stat(full_path).st_mtime
            if mtime > max_mtime:
                max_mtime = mtime
                max_dir = dirname
                max_file = fname
    return max_file

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
