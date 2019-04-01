import gzip
import os
import numpy as np

files = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}
dataset_dir = os.path.dirname(os.path.abspath(__file__))

def _load_img(filename):
    file_path = dataset_dir + "/" + filename
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)
    return data

def _load_label(filename):
    file_path = dataset_dir + "/" + filename
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T
    
def load_mnist(normalize=True, flatten=True, one_hot_label=True):
    dataset = {}
    for key in ('train_img', 'test_img'):
        dataset[key] = _load_img(files[key])

    for key in ('train_label', 'test_label'):
        dataset[key] = _load_label(files[key])

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    
    if one_hot_label:
        for key in ('train_label', 'test_label'):
            dataset[key] = _change_one_hot_label(dataset[key])
    
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    
    return ((dataset['train_img'], dataset['train_label']), 
                (dataset['test_img'], dataset['test_label']))

