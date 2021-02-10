import os
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

from lightfm import LightFM
from lightfm.evaluation import precision_at_k

import time

def make_comparison(train, model, val, k_param = 50, epochs_param = 10):
    print('running models')

    comparison = { 'time': [],
                 'val_precision': []
    }

    start_time = time.time()
    proc_start_time = time.process_time()

    model = model.fit(train, epochs = epochs_param, num_threads=1, verbose=True)

    val_precision = precision_at_k(model, val, k=k_param).mean()

    seconds = (time.time() - start_time)/60
    proc_time_elapsed = (time.time() - proc_start_time)/60

    comparison['time'].append([seconds,proc_time_elapsed])
    comparison['val_precision'].append(val_precision)
    print('done running models')
    print('DATASET_SIZE:',train.shape,'FIT TIME:',comparison['time'],'VAL P@K',comparison['val_precision'])

    return comparison

def make_matrix(interactions, val_interactions):

    ints = np.array(interactions)
    users = np.array(interactions['user_id'].index)
    books = np.array(interactions['book_id'].index)

    val_ints = np.array(val_interactions)
    val_user = np.array(val_interactions['user_id'].index)
    val_book = np.array(val_interactions['book_id'].index)

    n_users = len(np.unique(interactions['user_id']))
    n_items = len(np.unique(interactions['book_id']))

    train = csr_matrix((np.array(interactions['rating']), (users, books)))
    val = csr_matrix((np.array(val_interactions['rating']), (val_user, val_book)), shape=(len(users), len(books)))
    return train, val


def make_data(train_dataset, val_dataset):

    print('making data')
    interactions = train_dataset
    val_interactions = val_dataset

    interactions.columns = ['user_id', 'book_id', 'rating']
    val_interactions.columns = ['user_id', 'book_id', 'rating']

    # create sparse matrix, format required for lightfm
    train, val = make_matrix(interactions, val_interactions)

    print('done making data')

    return train, val

def main():
    # Load different file sizes, assumes they are each in their own folder
    print('loading data')

    # 0.1%, 0.5%, 1%, 2%, 5%
    interactions_001 = pd.read_csv('data_001/train_interactions_csv',header=None)
    val_interactions_001 = pd.read_csv('data_001/val_interactions_csv',header=None)

    interactions_005 = pd.read_csv('data_005/train_interactions_csv',header=None)
    val_interactions_005 = pd.read_csv('data_005/val_interactions_csv',header=None)

    interactions_01 = pd.read_csv('data_01/train_interactions_csv',header=None)
    val_interactions_01 = pd.read_csv('data_01/val_interactions_csv',header=None)

    interactions_02 = pd.read_csv('data_02/train_interactions_csv',header=None)
    val_interactions_02 = pd.read_csv('data_02/val_interactions_csv',header=None)

    interactions_05 = pd.read_csv('data_05/train_interactions_csv',header=None)
    val_interactions_05 = pd.read_csv('data_05/val_interactions_csv',header=None)

    # Start with 0.1% to 0.5%... not sure if computer can handle more
    datasets = [[interactions_001,val_interactions_001],
                [interactions_005,val_interactions_005],
                [interactions_01,val_interactions_01],
                [interactions_02,val_interactions_02]]

    print('done loading data')

    # define model parameters, rank = no_components
    model = LightFM(loss='warp',learning_rate=0.05, random_state=None, no_components=30)

    for d in datasets:

        print('running:',d[0].shape)

        train_matrix, val_matrix = make_data(d[0],d[1])

        comparison = make_comparison(train = train_matrix, model = model, val = val_matrix, k_param = 50, epochs_param = 10)

if __name__ == "__main__":

    main()
