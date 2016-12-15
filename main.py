import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from heapq import nsmallest
from sys import stdout
from random import shuffle
from datetime import datetime as dt
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, cpu_count
from threading import Thread
import json
import math
import concurrent
import sklearn

begin_time = dt.now().now()
def similarity(x, y, colloborative, content_based, verbose):
    # pull similarity from dataframe
    return similarity_score

def read_df_csv(fn,):
    if verbose:
        print("Reading in df {} from disk: {}".format(fn, get_time()))
    dir = 'data/processed/'
    fn = dir + fn
    df = pd.read_csv(fn)
    return csr


def read_sparse_file(fn):
    pass

def main():
    print("Initializing Script")
    print(begin_time)

    if verbose: print('Reading in means')

    if verbose: print('Running model on neighbors')
    knn = sklearn.neighbors.KNeighborsClassifie(n_neigbors=5, weights='uniform', algorithm='auto', leaf_size=3-, p=2,
                                          metric='minkowski',n_jobs=-2)
    knn.fit(train_data)
    knn.predict(test_data)

if __name__ == '__main__':
   main()
