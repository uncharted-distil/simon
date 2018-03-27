from faker import Factory
import numpy as np
import pandas as pd
import pprint
import json
from ColumnProcessor import ColType as ct
import ColumnProcessor as cp
import os.path
from random import randint
import random


class FakeDataDescriptor:
    def __init__(self):
        fake = Factory.create()
        fake.random.seed(1234)
        self.fake = fake
    methods = None
    with open(os.path.join(os.path.dirname(__file__),'types.json')) as data_file:
        methods = json.load(data_file)

    def show_example_data(self):
        for method in FakeDataDescriptor.methods:
            try:
                fake_func = getattr(self.fake, method)
                print("{0} - {1}.  Ex: {2}.... {3}".format(method,
                                                           FakeDataDescriptor.methods[method], str(fake_func()), str(fake_func())))
            except Exception as ex:
                print("{0} - Error: {1}".format(method, ex))


x = FakeDataDescriptor()
x.show_example_data()
