from dateutil.parser import parse
from enum import Enum
import pandas as pd


class ColType(Enum):
    string = 1
    date_time = 2
    integer = 3
    currency = 4
    decimal = 5
    boolean = 6


class IntChecker:
    my_type = ColType.integer

    def check(self, string):
        try:
            int(string)
            return True
        except Exception:
            return False


class DateChecker:
    my_type = ColType.date_time

    def check(self, string):
        try:
            parse(string)
            return True
        except Exception:
            return False


class FloatChecker:
    my_type = ColType.decimal

    def check(self, string):
        try:
            float(string)
            return True
        except Exception:
            return False


class BoolChecker:
    my_type = ColType.boolean

    def check(self, string):
        try:
            bool(string)
            return True
        except Exception:
            return False


class StringChecker:
    my_type = ColType.string

    def check(self, string):
        try:
            str(string)
            return True
        except Exception:
            return False


class ColProcessor:
    checkers = [BoolChecker(), IntChecker(), FloatChecker(),
                DateChecker(), StringChecker()]

    def __init__(self, uniques):
        self.uniques = uniques

    def get_type(self):
        outcomes = {}
        uniques = self.uniques
        for checker in ColProcessor.checkers:
            positives = 0.0
            for unique in uniques:
                if checker.check(unique):
                    positives += 1
            outcomes[checker.my_type] = positives / len(uniques)

        return outcomes


class MatrixProcessor:
    def __init__(self, matrix, headers):
        self.matrix = matrix
        self.headers = headers

    def process(self):
        pd_matrix = pd.DataFrame(self.matrix, columns=self.headers[:, 0])

        unique_value_dict = {}
        col_num = 0
        for column_name in pd_matrix.columns:
            column_mapping = {"actual": self.headers[col_num, 1]}

            uniques = pd_matrix[column_name].unique()
            col_processor = ColProcessor(uniques)
            col_type = col_processor.get_type()
            column_mapping["projection"] = col_type

            unique_value_dict[column_name] = column_mapping
            col_num += 1
        return unique_value_dict,  pd_matrix
        # print data_matrix[column_name].unique

        # data_matrix[column_name].map(unique_value_dict[column_name])
