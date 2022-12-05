import csv
import os
import sys


def read_csv_file(file_directory):
    with open(file_directory) as file:
        reader = csv.reader(file)
        try:
            return [line for line in reader]
        except csv.Error as error:
            sys.exit(f'{file_directory}{reader.line_num}{error}')



