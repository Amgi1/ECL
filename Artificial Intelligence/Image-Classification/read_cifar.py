import pickle
import numpy as np
import os
from math import floor


def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_cifar_batch(path):
    Dict = unpickle(path)
    data = np.array(Dict[b'data'], dtype=np.float32)
    labels = np.array(Dict[b'labels'], dtype=np.int64)
    return data, labels

def read_cifar(path):
    batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
    data_list, labels_list = [], []
    for file in batches:
        data_i, labels_i = read_cifar_batch(os.path.join(path, file))
        data_list.append(data_i)
        labels_list.append(labels_i)
    data = np.concatenate(data_list)
    labels = np.concatenate(labels_list)
    return data, labels


def split_dataset(data, labels, split):
    perm = np.random.permutation(data.shape[0])
    data = data[perm]
    labels = labels[perm]
    split_point = floor(split * data.shape[0])
    data_train = data[:split_point]
    data_test = data[split_point:]
    labels_train = labels[:split_point]
    labels_test = labels[split_point:]
    return data_train, labels_train, data_test, labels_test


if __name__ == "__main__":
    data, labels = read_cifar('data/cifar-10-batches-py/')
    print(split_dataset(data, labels, 0.5))