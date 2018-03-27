
import numpy as np
from Simon.DataGenerator import *
# reduces down a a x b matrix to (a*b) x 2 matrix where the second column
# is the column names from the header


class DataFlattener:

    @staticmethod
    def to_flat(data, header):

        flat = np.empty((data.shape[0] * data.shape[1], 2))
        for row_num in range(0, data.shape[0]):
            for col_num in range(0,  data.shape[1]):
                flat[row_num * col_num + col_num, 0] = data[row_num, col_num]
                flat[row_num * col_num + col_num, 1] = header[col_num, 1]

        return np.matrix(flat)

    @staticmethod
    def flatten(data, header):
        rows = data.shape[0]
        cols = data.shape[1]
        return data.reshape(rows * cols).tolist(), np.tile(header, rows).tolist()
    #(X_train, y_train), (X_test, y_test)

    @staticmethod
    def get_flat_data(data, header):

        # data_matrix = pd.DataFrame(data, columns=header)
        # print data_matrix
        # data_matrix.to_csv('test_data_{}.csv'.format(data_matrix.shape[0]))
        x_data, y_data = DataGenerator.flatten(data, header)
        size = len(x_data)

        #np.save('test_data_{}.csv'.format(size), np.hstack((x_data,y_data)))

        train = int(np.ceil(size * 2 / 3))
        test = int(np.ceil(size / 3))
        return (x_data[:train], y_data[:train]), (x_data[train + 1:], y_data[train + 1:])
