import unittest
from DataLengthRandomizer import *
import numpy as np

class Test_DataLengthRandomizer(unittest.TestCase):
    def randomize_dups_test(self):
        x = np.random.rand(5,10, 50)
        y = DataLengthRandomizer.randomize_dups(x, 3, 8)
        assert True


if __name__ == '__main__':
    unittest.main()
