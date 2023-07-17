import json
import csv
from datetime import datetime

from os.path import isfile

def retrieve_or_save_data(data_id, _data, str_format = False):
    path_to_data_file = f'cache/data/{data_id}.csv'

    if isfile(path_to_data_file):
        data = []
        with open(path_to_data_file) as csv_file:
            csv_reader = csv.reader(csv_file)
            for i, row in enumerate(csv_reader):
                if i == 0:
                    continue
                if str_format:
                    date = row[0]
                else:
                    date = datetime.strptime(row[0], '%Y-%m-%d')
                value = float(row[1])
                datum = {}
                datum['date'] = date
                datum['value'] = value
                data.append(datum)

    else:
        with open(path_to_data_file, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['date', 'value'])
            data = []
            for datum in _data:
                csv_writer.writerow([datum['date'][:10], datum['value']])
                if str_format:
                    datum['date'] = datum['date'][:10]
                else:
                    datum['date'] = datetime.strptime(datum['date'][:10], '%Y-%m-%d')
                data.append(datum)
    return data

def retrieve_or_save_metadata(metadata_id, metadata):
    path_to_metadata_file = f'cache/metadata/{metadata_id}.csv'