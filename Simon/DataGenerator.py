from faker import Factory
import numpy as np
import pandas as pd
import pprint
import json
from Simon.ColumnProcessor import ColType as ct
import Simon.ColumnProcessor as cp
import os.path
from random import randint
import random


class FakeDataCreator:
    methods = None
    with open(os.path.join(os.path.dirname(__file__),'types.json')) as data_file:
        methods = json.load(data_file)
    filtered_dict = {k: v for k, v in methods.items() if len(v)
                     > 0 and not v in ['']}

    def __init__(self):
        fake = Factory.create()
        fake.random.seed(1234)
        self.fake = fake

    def get_generator(self):
        fake_func = None
        column_name = ""
        valid = False
        fake = self.fake
        column_category = ""
        while not valid:
            try:
                #func_idx = randint(dir(fake).index('address'), len(dir(fake)))
                #column_name = dir(fake)[func_idx]
                column_name = random.choice(
                    list(FakeDataCreator.filtered_dict.keys()))
                fake_func = getattr(fake, column_name)
                valid = (not column_name in ['binary']) and callable(fake_func) and not isinstance(
                    fake_func(), (list, tuple, np.ndarray)) and str(fake_func())
                #print("{0} - {1}".format(column_name, str(fake_func())))
            except Exception as ex:
                print (ex)
                continue
        return fake_func, column_name

    def map_column_names_to_types(self, column_names):
        # print(column_names)
        filtered_dict = FakeDataCreator.filtered_dict
        return np.array([filtered_dict[column_name] if (column_name in filtered_dict) else None for column_name in column_names])


class DataGenerator:
    # this method isn't getting used currently..  it tries to create a fixed
    # dataset and describe it.
    # this may be beneficial to help us create predictable results and while
    # doing data encrichment
    @staticmethod
    def gen_data(count):
        path = "data/"
        matrix_name = path + "matrix_{0}.npy".format(count)
        header_name = path + "header_{0}.npy".format(count)
        if os.path.isfile(matrix_name) and os.path.isfile(header_name):
            print("loading data")
            return np.load(matrix_name), np.load(header_name)
        print("creating data")
        fake = Factory.create()
        fake.random.seed(1234)

        data = []
        for _ in range(0, count):
            first = fake.first_name()
            last = fake.last_name()
            address = fake.address()
            date = fake.date_time().strftime("%c")
            is_active = fake.boolean()
            cc_num = fake.credit_card_number()
            phone = fake.phone_number()
            ssn = fake.ssn()
            email = fake.safe_email()
            balance = fake.random_number()
            data.append([first, last, last + ", " + first, address,
                         date, is_active, cc_num, phone, ssn, email, balance])

        headers = np.array([["FIRST", ct.string], ["LAST_NAME", ct.string], ["NAME", ct.string], ["ADDRESS", ct.string], ["DATE", ct.date_time], [
                           "IsActive", ct.boolean], ["CC#", ct.string], ["Phone", ct.string], ["SS#", ct.string], ["Email", ct.string], ["Balance", ct.decimal]])
        #headers = headers[np.newaxis,:]
        matrix = np.matrix(data)
        with open(matrix_name, 'wb') as matrix_file:
            np.save(matrix_file, matrix)
        with open(header_name, 'wb') as header_file:
            np.save(header_file, headers)

        return matrix, headers

    @staticmethod
    def gen_col_data(data_creator, rows):
        fake_func, column_name = data_creator.get_generator()
        #print ("fake: {0}, col: {1}".format(fake_func, column_name))
        ret_val = np.empty((rows,), dtype="object")
        for i in range(rows):
            try:
                val = fake_func()
                # TODO: utf-8 is probably not acceptable for
                # internationalization
                ret_val[i] = str(val)  # repr.encode('utf-8')
            except Exception:
                print("error: {0}".format(fake_func()))
        return ret_val, column_name
    # generates a dataset given a certain size
    # will save it to data/ when complete

    @staticmethod
    def gen_test_data(shape, try_reuse_data):
        str_shape = 'x'.join([str(x) for x in shape])
        data_path = "data/"
        if not os.path.isdir(data_path):
            os.makedirs(data_path)
        matrix_name = data_path + "matrix_{0}.npy".format(str_shape)
        header_name = data_path + "header_{0}.npy".format(str_shape)
        data_creator = FakeDataCreator()
        if try_reuse_data and os.path.isfile(matrix_name) and os.path.isfile(header_name):
            print("loading data")
            ret_val = np.load(matrix_name)
            temp_header = data_creator.map_column_names_to_types(np.load(header_name))
            ret_header = np.asarray([temp_header[i] for i in np.arange(temp_header.shape[0]) if not temp_header[i]==None])
            # print("ret_header shape:")
            # print(ret_header.shape)
            del_idx = np.zeros(ret_val.shape[1]-ret_header.shape[0], dtype='object')
            k = 0
            for j in np.arange(temp_header.shape[0]):
                if(temp_header[j]==None):
                    # print("deleting column %d"%j)
                    del_idx[k] = j
                    k = k+1
            # print("del_idx:")
            # print(del_idx)
            ret_val = np.delete(ret_val,del_idx,axis=1) #delete redundant faker columns from data...
            # print("ret_header")
            # print(ret_header)
            # print(ret_header.shape)
            # print(ret_val.shape)
            return ret_val, ret_header 
        print("creating data")
        ret_val = np.empty(shape, dtype="object")
        headers = np.empty(shape[1], dtype="object")

        for j in range(shape[1]):
            col_data, column_name = DataGenerator.gen_col_data(data_creator, shape[0])
            # fake_func, column_name = data_creator.get_generator()
            headers[j] = column_name
            ret_val[:, j] = col_data
            # for i in range(shape[0]):
            #     try:
            #         val= fake_func()
            #         #TODO: utf-8 is probably not acceptable for
            #         internationalization
            #         ret_val[i,j] = str(val)#repr.encode('utf-8')
            #     except Exception:
            #         print "error: {0}".format(fake_func())

        with open(matrix_name, 'wb') as matrix_file:
            np.save(matrix_file, ret_val)
        with open(header_name, 'wb') as header_file:
            np.save(header_file, headers)
        
        temp_header = data_creator.map_column_names_to_types(headers)
        ret_header = np.asarray([temp_header[i] for i in np.arange(temp_header.shape[0]) if not temp_header[i]==None])
        del_idx = np.zeros(ret_val.shape[1]-ret_header.shape[0], dtype='object')
        k = 0
        for j in np.arange(temp_header.shape[0]):
            if(temp_header[j]==None):
                del_idx[k] = j #deleting column j
                k = k+1
        ret_val = np.delete(ret_val,del_idx,axis=1) #delete redundant faker columns from data...
        
        return ret_val, ret_header

    @staticmethod
    def add_col_nulls(data, col_idx, percent=.05):
        count = int(data.shape[0] * percent)
        rows = [randint(0, data.shape[0] - 1) for p in range(count)]
        nulls = ['NULL', 'N/A', 'n/a', 'na', 'None', 'Nothing', '', '-']
        data[rows, col_idx] = random.choice(nulls)

    @staticmethod
    def add_nulls_uniform(data, percent=.05):
        for idx in range(data.shape[1]):
            DataGenerator.add_col_nulls(data,idx, percent)


if __name__ == '__main__':
    from Encoder import *
    import ColumnProcessor as cp
    data, header = DataGenerator.gen_test_data((100, 50))
    encoded_data = Encoder.encode_matrix(data, StringToIntArrayEncoder())

    processor = cp.MatrixProcessor(data, header)
    col_types, matrix = processor.process()
    print(matrix)
    #pd_matrix = pd.DataFrame(data, columns=header[:,0])
    #type_provider = cp.ColProcessor(pd_matrix)
