#from Simon.DataGenerator import *
#from Simon.Encoder import *
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Dropout, MaxPooling1D, Convolution1D
from tensorflow.keras.layers import LSTM, Lambda, Masking, Embedding, TimeDistributed, BatchNormalization, concatenate
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import tensorflow.keras.callbacks as callbacks

import re
import os
import time
import pickle
import logging

logger = logging.getLogger(__name__)

# record history of training
class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('binary_accuracy'))

class Simon:
    def __init__(self,encoder):
        self.encoder = encoder

    def binarize(self,x, sz=71):
        return tf.cast(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1), dtype=tf.float32)

    def clear_session(self):
        K.clear_session()

        return
    
    def custom_multi_label_accuracy(y_true, y_pred):
        # need some threshold-specific rounding code here, presently only for 0.5 thresh.
        return K.mean(K.round(np.multiply(y_true,y_pred)),axis=0)
    
    def eval_binary_accuracy(self,y_test, y_pred):
        correct_indices = y_test==y_pred
        all_correct_predictions = np.zeros(y_test.shape)
        all_correct_predictions[correct_indices] = 1
        return np.mean(all_correct_predictions),np.mean(all_correct_predictions, axis=0),all_correct_predictions
    
    def eval_confusion(self,y_test, y_pred):
        wrong_indices = y_test!=y_pred
        all_wrong_predictions = np.zeros(y_test.shape)
        all_wrong_predictions[wrong_indices] = 1
        return np.mean(all_wrong_predictions),np.mean(all_wrong_predictions, axis=0),all_wrong_predictions
    
    def eval_false_positives(self,y_test, y_pred):
        false_positive_matrix = np.zeros((y_test.shape[1],y_test.shape[1]))
        false_positives = np.multiply(y_pred,1-y_test)
        for i in np.arange(y_test.shape[0]):
            for j in np.arange(y_test.shape[1]) :
                if(false_positives[i,j]==1): #positive label for ith sample and jth predicted category
                    for k in np.arange(y_test.shape[1]):
                        if(y_test[i,k]==1): #positive label for ith sample and kth true category
                            false_positive_matrix[j,k] +=1
        return np.sum(false_positive_matrix),np.sum(false_positive_matrix, axis=0),false_positive_matrix
    
    def eval_ROC_metrics(self,y_test, y_pred):
        true_positive = y_pred*y_test
        true_negative = (1-y_pred)*(1-y_test)
        false_positive = y_pred*(1-y_test)
        false_negative = (1-y_pred)*y_test

        return np.sum(true_positive,axis=0),np.sum(true_negative,axis=0),np.sum(false_positive,axis=0),np.sum(false_negative,axis=0)
    
    
    def binarize_outshape(self,in_shape):
        return in_shape[0], in_shape[1], 71

    def max_1d(x):
        return K.max(x, axis=1)

    def striphtml(html):
        p = re.compile(r'<.*?>')
        return p.sub('', html)

    def clean(s):
        return re.sub(r'[^\x00-\x7f]', r'', s)

    def setup_test_sets(self,X, y):
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

    def generate_model(self, max_len, max_cells, category_count, activation='sigmoid', training = False):
        filter_length = [1, 3, 3]
        nb_filter = [40, 200, 1000]
        pool_size = 2
        # document input
        document = Input(shape=(max_cells, max_len), dtype='int64')
        # sentence input
        in_sentence = Input(shape=(max_len,), dtype='int64')
        # char indices to one hot matrix, 1D sequence to 2D
        embedded = Lambda(self.binarize, output_shape=self.binarize_outshape)(in_sentence)
        # embedded: encodes sentence
        for i in range(len(nb_filter)):
            embedded = Convolution1D(nb_filter[i],
                                     filter_length[i],
                                     activation='relu',
                                     kernel_initializer='glorot_normal')(embedded)

            embedded = Dropout(0.1)(embedded)
            embedded = MaxPooling1D(pool_size=pool_size)(embedded)
        
        forward_sent = LSTM(256, return_sequences=False, dropout=0.2,
                        recurrent_dropout=0.2)(embedded, training = training)
        backward_sent = LSTM(256, return_sequences=False, dropout=0.2,
                        recurrent_dropout=0.2, go_backwards=True)(embedded, training = training)

        sent_encode = concatenate([forward_sent, backward_sent], axis=-1)
        sent_encode = Dropout(0.3)(sent_encode)
        # sentence encoder

        encoder = Model(inputs=in_sentence, outputs=sent_encode)

        #logger.debug(encoder.summary())
        encoded = TimeDistributed(encoder)(document)

        # encoded: sentences to bi-lstm for document encoding
        forwards = LSTM(128, return_sequences=False, dropout=0.2,
                        recurrent_dropout=0.2)(encoded, training = training)
        backwards = LSTM(128, return_sequences=False, dropout=0.2,
                        recurrent_dropout=0.2, go_backwards=True)(encoded, training = training)

        merged = concatenate([forwards, backwards], axis=-1)
        output = Dropout(0.3)(merged)
        output = Dense(128, activation='relu')(output)
        output = Dropout(0.3)(output)
        output = Dense(category_count, activation=activation)(output)
        # output = Activation('softmax')(output)
        model = Model(inputs=document, outputs=output)

        return model
        
    def generate_transfer_model(self,max_len, max_cells, category_count_prior, category_count_post, checkpoint, checkpoint_dir,activation='sigmoid'):
        filter_length = [1, 3, 3]
        nb_filter = [40, 200, 1000]
        pool_size = 2
        # document input
        document = Input(shape=(max_cells, max_len), dtype='int64')
        # sentence input
        in_sentence = Input(shape=(max_len,), dtype='int64')
        # char indices to one hot matrix, 1D sequence to 2D
        embedded = Lambda(self.binarize, output_shape=self.binarize_outshape)(in_sentence)
        # embedded: encodes sentence
        for i in range(len(nb_filter)):
            embedded = Convolution1D(nb_filter[i],
                                     filter_length[i],
                                     activation='relu',
                                     kernel_initializer='glorot_normal')(embedded)

            embedded = Dropout(0.1)(embedded)
            embedded = MaxPooling1D(pool_size=pool_size)(embedded)

        forward_sent = LSTM(256, return_sequences=False, dropout=0.2,
                        recurrent_dropout=0.2)(embedded)
        backward_sent = LSTM(256, return_sequences=False, dropout=0.2,
                        recurrent_dropout=0.2, go_backwards=True)(embedded)

        sent_encode = concatenate([forward_sent, backward_sent], axis=-1)
        sent_encode = Dropout(0.3)(sent_encode)
        # sentence encoder

        encoder = Model(inputs=in_sentence, outputs=sent_encode)

        #logger.debug(encoder.summary())
        encoded = TimeDistributed(encoder)(document)

        # encoded: sentences to bi-lstm for document encoding
        forwards = LSTM(128, return_sequences=False, dropout=0.2,
                        recurrent_dropout=0.2)(encoded)
        backwards = LSTM(128, return_sequences=False, dropout=0.2,
                        recurrent_dropout=0.2, go_backwards=True)(encoded)

        merged = concatenate([forwards, backwards], axis=-1)
        output_pre = Dropout(0.3)(merged)
        output_pre = Dense(128, activation='relu')(output_pre)
        output_pre = Dropout(0.3)(output_pre)
        output = Dense(category_count_prior, activation=activation)(output_pre)
        model = Model(inputs=document, outputs=output)
    
        # having built model of prior size, load weights
        self.load_weights(checkpoint, None, model, checkpoint_dir)
        # having loaded weights,rebuild model using new category_count in last layer
        output = Dense(category_count_post, activation='sigmoid')(output_pre)
        model = Model(inputs=document, outputs=output)
        # retrain the last layer
        for layer in model.layers[:8]:
            layer.trainable = False
        model.layers[8].trainable = True

        return model

    def generate_feature_model(self, max_len, max_cells, category_count, checkpoint_dir, config, DEBUG = True):
        '''
            Generate SIMON feature model which produces 128-d SIMON features as last layer
        '''
        filter_length = [1, 3, 3]
        nb_filter = [40, 200, 1000]
        pool_size = 2
        # document input
        document = Input(shape=(max_cells, max_len), dtype='int64')
        # sentence input
        in_sentence = Input(shape=(max_len,), dtype='int64')
        # char indices to one hot matrix, 1D sequence to 2D
        embedded = Lambda(self.binarize, output_shape=self.binarize_outshape)(in_sentence)
        # embedded: encodes sentence
        for i in range(len(nb_filter)):
            embedded = Convolution1D(nb_filter[i],
                                     filter_length[i],
                                     activation='relu',
                                     kernel_initializer='glorot_normal')(embedded)

            embedded = Dropout(0.1)(embedded)
            embedded = MaxPooling1D(pool_size=pool_size)(embedded)

        forward_sent = LSTM(256, return_sequences=False, dropout=0.2,
                        recurrent_dropout=0.2)(embedded)
        backward_sent = LSTM(256, return_sequences=False, dropout=0.2,
                        recurrent_dropout=0.2, go_backwards=True)(embedded)

        sent_encode = concatenate([forward_sent, backward_sent], axis=-1)
        sent_encode = Dropout(0.3)(sent_encode)
        # sentence encoder

        encoder = Model(inputs=in_sentence, outputs=sent_encode)

        #logger.debug(encoder.summary())
        encoded = TimeDistributed(encoder)(document)

        # encoded: sentences to bi-lstm for document encoding
        forwards = LSTM(128, return_sequences=False, dropout=0.2,
                        recurrent_dropout=0.2)(encoded)
        backwards = LSTM(128, return_sequences=False, dropout=0.2,
                        recurrent_dropout=0.2, go_backwards=True)(encoded)

        merged = concatenate([forwards, backwards], axis=-1)
        output_pre = Dropout(0.3)(merged)
        output_pre = Dense(128, activation='relu', name='features')(output_pre)
        output = Dropout(0.3)(output_pre)
        output = Dense(category_count, activation='softmax')(output)
        model = Model(inputs=document, outputs=output)
        self.load_weights(config['checkpoint'],config,model,checkpoint_dir)

        '''GENERATE MODEL FOR INTERMEDIATE LAYER'''
        intermediate_model = Model(inputs=model.input, outputs=model.get_layer('features').output)
        return intermediate_model

    def plot_loss(self,history):
        import matplotlib.pyplot as plt
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

    def train_model(self,batch_size, checkpoint_dir, model, nb_epoch, data):
        logger.debug("starting learning")
    
        check_cb = callbacks.ModelCheckpoint(checkpoint_dir + "text-class" + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                    monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        earlystop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')

        tbCallBack = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0,
                                    embeddings_layer_names=None, embeddings_metadata=None)
        loss_history = LossHistory()
        history = model.fit(data.X_train, data.y_train, validation_data=(data.X_cv_test, data.y_cv_test), batch_size=batch_size,
                    nb_epoch=nb_epoch, shuffle=True, callbacks=[earlystop_cb, check_cb, loss_history, tbCallBack])
    
        logger.debug('losses: ')
        logger.debug(history.history['loss'])
        logger.debug('accuracies: ')
        logger.debug(history.history['val_binary_accuracy'])
        return history

    def evaluate_model(self,max_cells, model, data, encoder, p_threshold):
        logger.debug("Starting predictions:")
    
        start = time.time()
        scores = model.evaluate(data.X_test, data.y_test, verbose=0)
        end = time.time()
        logger.debug("Accuracy: %.2f%% \n Time: {0}s \n Time/example : {1}s/ex".format(
            end - start, (end - start) / data.X_test.shape[0]) % (scores[1] * 100))
        
        probabilities = model.predict(data.X_test, verbose=1)
        y_pred = np.where(y > p_threshold, 1, 0).astype(int)

        logger.debug("'Binary' accuracy ((TP+TN)/total) sample number is:")
        logger.debug(self.eval_binary_accuracy(data.y_test,y_pred)[0])
        logger.debug("'Binary' confusion ((FP+FN)/total) sample number is:")
        logger.debug(self.eval_confusion(data.y_test,y_pred)[0])
        logger.debug("False Positive sample number is:")
        logger.debug(self.eval_false_positives(data.y_test,y_pred)[0])
        TP,TN,FP,FN = self.eval_ROC_metrics(data.y_test, y_pred)
        logger.debug("Precision is:")
        logger.debug(np.sum(TP)/(np.sum(TP)+np.sum(FP)))
        logger.debug("Recall is:")
        logger.debug(np.sum(TP)/(np.sum(TP)+np.sum(FN)))
        logger.debug("F1 score is:")
        logger.debug(2*np.sum(TP)/(2*np.sum(TP)+np.sum(FP)+np.sum(FN)))


        return encoder.reverse_label_encode(probabilities,p_threshold)

    def resolve_file_path(self,filename, dir):
        if os.path.isfile(str(filename)):
            return str(filename)
        elif os.path.isfile(str(dir + str(filename))):
            return dir + str(filename)

    def load_weights(self,checkpoint_name, config, model, checkpoint_dir):
        if config and not checkpoint_name:
            checkpoint_name = config['checkpoint']
        if checkpoint_name:
            checkpoint_path = self.resolve_file_path(checkpoint_name, checkpoint_dir)
            #logger.debug("Checkpoint : %s" % str(checkpoint_path))
            model.load_weights(checkpoint_path)

    def save_config(self,execution_config, checkpoint_dir):
        filename = ""
        if not execution_config["checkpoint"] is None:
            filename = execution_config["checkpoint"].rsplit( ".", 1 )[ 0 ] + ".pkl"
        else:
            filename = time.strftime("%Y%m%d-%H%M%S") + ".pkl"
        with open(checkpoint_dir + filename, 'wb') as output:
            pickle.dump(execution_config, output, pickle.HIGHEST_PROTOCOL)

    def load_config(self,execution_config_path, dir):
        execution_config_path = self.resolve_file_path(execution_config_path, dir)
        config = pickle.load( open( execution_config_path, "rb" ) )
        self.encoder = config['encoder']
        return config

    def get_best_checkpoint(self,checkpoint_dir):
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

    ###### WHAT FOLLOWS IS FCHOLLET'S MULTI-GPU CODE #######
    def _get_available_devices(self):
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos]


    def multi_gpu_model(self, model, gpus):
        """Replicates a model on different GPUs.
        Specifically, this function implements single-machine
        multi-GPU data parallelism. It works in the following way:
        - Divide the model's input(s) into multiple sub-batches.
        - Apply a model copy on each sub-batch. Every model copy
            is executed on a dedicated GPU.
        - Concatenate the results (on CPU) into one big batch.
        E.g. if your `batch_size` is 64 and you use `gpus=2`,
        then we will divide the input into 2 sub-batches of 32 samples,
        process each sub-batch on one GPU, then return the full
        batch of 64 processed samples.
        This induces quasi-linear speedup on up to 8 GPUs.
        This function is only available with the TensorFlow backend
        for the time being.
        # Arguments
            model: A Keras model instance. To avoid OOM errors,
                this model could have been built on CPU, for instance
                (see usage example below).
            gpus: Integer >= 2, number of on GPUs on which to create
                model replicas.
        # Returns
            A Keras `Model` instance which can be used just like the initial
            `model` argument, but which distributes its workload on multiple GPUs.
        # Example
        ```python
            import tensorflow as tf
            from keras.applications import Xception
            num_samples = 1000
            height = 224
            width = 224
            num_classes = 1000
            # Instantiate the base model
            # (here, we do it on CPU, which is optional).
            with tf.device('/cpu:0'):
                model = Xception(weights=None,
                                input_shape=(height, width, 3),
                                classes=num_classes)
            # Replicates the model on 8 GPUs.
            # This assumes that your machine has 8 available GPUs.
            parallel_model = multi_gpu_model(model, gpus=8)
            parallel_model.compile(loss='categorical_crossentropy',
                                optimizer='rmsprop')
            # Generate dummy data.
            x = np.random.random((num_samples, height, width, 3))
            y = np.random.random((num_samples, num_classes))
            # This `fit` call will be distributed on 8 GPUs.
            # Since the batch size is 256, each GPU will process 32 samples.
            parallel_model.fit(x, y, epochs=20, batch_size=256)
        ```
        """
        if K.backend() != 'tensorflow':
            raise ValueError('`multi_gpu_model` is only available '
                            'with the TensorFlow backend.')
        if gpus <= 1:
            raise ValueError('For multi-gpu usage to be effective, '
                            'call `multi_gpu_model` with `gpus >= 2`. '
                            'Received: `gpus=%d`' % gpus)

        target_devices = ['/device:CPU:0'] + ['/device:GPU:%d' % i for i in range(gpus)]
        available_devices = self._get_available_devices()
        for device in target_devices:
            if device not in available_devices:
                raise ValueError(
                    'To call `multi_gpu_model` with `gpus=%d`, '
                    'we expect the following devices to be available: %s. '
                    'However this machine only has: %s. '
                    'Try reducing `gpus`.' % (gpus,
                                            target_devices,
                                            available_devices))

        def get_slice(data, i, parts):
            shape = tf.shape(input=data)
            batch_size = shape[:1]
            input_shape = shape[1:]
            step = batch_size // parts
            if i == gpus - 1:
                size = batch_size - step * i
            else:
                size = step
            size = tf.concat([size, input_shape], axis=0)
            stride = tf.concat([step, input_shape * 0], axis=0)
            start = stride * i
            return tf.slice(data, start, size)

        all_outputs = []
        for i in range(len(model.outputs)):
            all_outputs.append([])

        # Place a copy of the model on each GPU,
        # each getting a slice of the inputs.
        for i in range(gpus):
            with tf.device('/device:GPU:%d' % i):
                with tf.compat.v1.name_scope('replica_%d' % i):
                    inputs = []
                    # Retrieve a slice of the input.
                    for x in model.inputs:
                        input_shape = tuple(x.get_shape().as_list())[1:]
                        slice_i = Lambda(get_slice,
                                        output_shape=input_shape,
                                        arguments={'i': i,
                                                    'parts': gpus})(x)
                        inputs.append(slice_i)

                    # Apply model on slice
                    # (creating a model replica on the target device).
                    outputs = model(inputs)
                    if not isinstance(outputs, list):
                        outputs = [outputs]

                    # Save the outputs for merging back together later.
                    for o in range(len(outputs)):
                        all_outputs[o].append(outputs[o])

        # Merge outputs on CPU.
        with tf.device('/device:CPU:0'):
            merged = []
            for outputs in all_outputs:
                merged.append(concatenate(outputs,
                                        axis=0))
            return Model(model.inputs, merged)

    def tune_ROC_metrics(self,max_cells, model, data, encoder,p_thresholds):
        logger.debug("Starting to compute ROC metrics...")

        start = time.time()
        scores = model.evaluate(data.X_test, data.y_test, verbose=0)
        end = time.time()
        logger.debug("Initial Accuracy: %.2f%% \n Time: {0}s \n Time/example : {1}s/ex".format(
            end - start, (end - start) / data.X_test.shape[0]) % (scores[1] * 100))

        probabilities = model.predict(data.X_test, verbose=1)

        Categories = encoder.categories
        logger.debug("The fixed categories are:")
        logger.debug(Categories)

        # evaluate ROC metrics for a number of p_threholds
        TPR_arr = np.zeros((p_thresholds.shape[0],1))
        FPR_arr = np.zeros((p_thresholds.shape[0],1))
        i = 0
        for p_threshold in p_thresholds:
            y_pred = np.where(y > p_threshold, 1, 0).astype(int)
            TP,TN,FP,FN = self.eval_ROC_metrics(data.y_test, y_pred)
            TPR_arr[i,:]= np.sum(TP)/(np.sum(TP)+np.sum(FN))
            FPR_arr[i,:]= np.sum(FP)/(np.sum(FP)+np.sum(TN))
            i = i+1

        # save data for analysis later?
        # np.save("TP",TP)
        # np.save("TN",TN)
        # np.save("FP",FP)
        # np.save("FN",FN)
        # np.save("TPR_arr",TPR_arr)
        # np.save("FPR_arr",FPR_arr)

        return TPR_arr,FPR_arr
