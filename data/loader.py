import os
from keras import backend as K
import numpy as np
from src.processing.folders import Folders
K.set_image_data_format('channels_last')

class DataLoader(object):
    @classmethod
    def load(cls, dataset='nucleus', set='training', records = -1, separate=True):
        data_path = Folders.data_folder() + '{0}-{1}.npz'.format(dataset, set)
        if os.path.isfile(data_path):
            raw = np.load(data_path, mmap_mode='r')
        if records > 0:
            # additional logic for efficient caching of small subsets of records
            raw_trunc = Folders.data_folder() + '{0}-{1}-n{2}.npz'.format(dataset, set, records)
            if os.path.isfile(raw_trunc):
                raw_n = np.load(raw_trunc, mmap_mode='r')
                return raw_n['data'], raw_n['labels']
            else:
                data, labels = raw['data'][0:records, ...], raw['labels'][0:records, ...]
                np.savez(raw_trunc, data=data, labels=labels)
                return data, labels
        else:
            return raw['data'], raw['labels']


    @classmethod
    def load_training(cls, dataset='nucleus', records=-1, separate=True):
        return DataLoader.load(dataset=dataset, set='training', records=records, separate=separate)

    @classmethod
    def load_testing(cls, dataset='nucleus', records=-1, separate=True):
        return DataLoader.load(dataset=dataset, set='test', records=records, separate=separate)

    @classmethod
    def batch_data(cls, train_data, train_labels, batch_size):
        """ Simple sequential chunks of data """
        for batch in range(int(np.ceil(train_data.shape[0] / batch_size))):
            start = batch_size * batch
            end = start + batch_size
            if end > train_data.shape[0]:
                yield train_data[-batch_size:, ...], \
                        train_labels[-batch_size:, ...]
            else:
                yield train_data[start:end, ...], \
                        train_labels[start:end, ...]

# Load just 64 training records
# DataLoader.load_training(records=64)