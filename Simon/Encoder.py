import json
import pandas
import os.path
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from Simon.LengthStandardizer import *
import logging

class StringToIntArrayEncoder:
    def encode(self, string, char_count):
        return [ord(c) for c in string.ljust(char_count)[:char_count]]


class Encoder:
    def __init__(self,categories):
        self._char_indicies = {}
        self._indices_char = {}
        self.cur_max_cells = 0
        self._encoder = LabelEncoder()
        self._multi_encoder = MultiLabelBinarizer()
        self.categories = categories


    def process(self, raw_data, max_cells):
        chars = {}
        for doc in raw_data:
            for cell in doc:
                for c in str(cell):
                    if not c in chars:
                        chars[c] = True
        #this line is sounding increasingly unnecesary, probably should just always leave it capped to 500
        self.cur_max_cells = min(raw_data.shape[1], max_cells)
        chars = sorted(list(chars))

        self.char_indices = dict((c, i) for i, c in enumerate(chars))
        self._indices_char = dict((i, c) for i, c in enumerate(chars))
  
    def encode_matrix(self, matrix):
        chars = 20
        new_matrix = np.zeros((matrix.shape[0], matrix.shape[1], chars))
        for row in range(0, matrix.shape[0]):
            new_array = []
            for col in range(0, matrix.shape[1]):
                new_matrix[row, col] = self._encoder.encode(matrix[row, col], chars)
                
        return new_matrix

    def encode_data(self, raw_data, header, max_len):
        
        X = np.ones((raw_data.shape[0], self.cur_max_cells,
                     max_len), dtype=np.int64) * -1
        y = self.label_encode(header)
        
        # track unencoded chars
        unencoded_chars = None
        if not os.path.isfile('unencoded_chars.json'):
            open('unencoded_chars.json','w').close()
            unencoded_dict = {}
        else:
            with open('unencoded_chars.json') as data_file:
                unencoded_chars = json.load(data_file)
            unencoded_dict = {k: v for k, v in unencoded_chars.items() if not v in ['']}

        for i, column in enumerate(raw_data):
            for j, cell in enumerate(column):
                if(j < self.cur_max_cells):
                    string = cell[-max_len:].lower()
                    for k, char in enumerate(string):
                        try:
                            X[i, j, (max_len - 1 - k)] = self.char_indices[char]
                        except Exception:
                            #X[i, j, (max_len - 1 - k)] = self.char_indices[' ']
                            if char in unencoded_dict.keys():
                                unencoded_dict[char] = unencoded_dict[char]+1
                            else:
                                unencoded_dict[char] = 1
        with open('unencoded_chars.json','w') as out_file:
            json.dump(unencoded_dict, out_file)
        print("X shape: {0}\ny shape: {1}".format(X.shape, y.shape))
        
        return X, y

    def decode_matrix(self, X):
        ret_val = np.empty((X.shape[0], X.shape[1]), dtype='object')
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                string = ""
                # start at the end since we reversed it
                for k in range(X.shape[2] - 1, -1, -1):
                    if X[i, j, k] < 0:
                        continue
                    string = string + self._indices_char[X[i, j, k]]
                ret_val[i, j] = string
        return ret_val

    def label_encode(self, Y):
        # encode class values as integers
        categories = self.categories
        self._multi_encoder.fit([categories])

        # convert integers to dummy variables (i.e.  one hot encoded)
        multi_encoder = self._multi_encoder
                
        ret_val = multi_encoder.transform(Y)
        
        return ret_val

    def reverse_label_encode(self, y, p_threshold):
        
        multi_encoder = self._multi_encoder
        
        # convert predictions to binary integers
        y_pred = np.where(y > p_threshold, 1, 0).astype(int)
        
        # extract label prediction probabilities
        label_probs = [[prob for prob, pred in zip(prob_list, pred_list) if pred == 1] 
            for prob_list, pred_list in zip(np.array(y), y_pred)]

        # extract labels
        labels = multi_encoder.inverse_transform(y_pred)

        return labels, label_probs
        
    def encodeDataFrame(self, df: pandas.DataFrame):
        
        nx,ny = df.shape
        
        out = DataLengthStandardizerRaw(df,self.cur_max_cells)
        
        raw_data = np.char.lower(np.transpose(out).astype('U'))
        
        return self.x_encode(raw_data, 20)  
        
        
    def x_encode(self, raw_data, max_len):

        X = np.ones((raw_data.shape[0], self.cur_max_cells,
                     max_len), dtype=np.int64) * -1        
                
        # track unencoded chars
        unencoded_chars = None
        if not os.path.isfile('unencoded_chars.json'):
            unencoded_dict = {}
        else:
            with open('unencoded_chars.json') as data_file:
                unencoded_chars = json.load(data_file)
            unencoded_dict = {k: v for k, v in unencoded_chars.items() if not v in ['']}

        for i, column in enumerate(raw_data):
            for j, cell in enumerate(column):
                if(j < self.cur_max_cells):
                    string = cell[-max_len:].lower()
                    for k, char in enumerate(string):
                        try:
                            X[i, j, (max_len - 1 - k)] = self.char_indices[char]
                        except Exception:
                            X[i, j, (max_len - 1 - k)] = self.char_indices[' '];
                            if char in unencoded_dict.keys():
                                unencoded_dict[char] = unencoded_dict[char]+1
                            else:
                                unencoded_dict[char] = 1
        with open('unencoded_chars.json','w') as out_file:
            json.dump(unencoded_dict, out_file)
        
        return X