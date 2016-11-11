import matplotlib.pyplot as plt
from heapq import nsmallest
from Amazon_review import Amazon_review
from sys import stdout
from random import shuffle
from datetime import datetime as dt
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, cpu_count
from threading import Thread
import json
import math
import concurrent

begin_time = dt.now().now()
def similarity(x, y, colloborative, content_based, verbose):
    if verbose:
        print("finding similarity")
    similarity_score = 0
    if colloborative:
        # do pearson similarity
        pass
    if content_based:
        # do cosine similarity
        pass
    pass
    return similarity_score

def find_row_means()
def main():
    print("Initializing Script")
    print(begin_time)
    sentiWord_d = read_in_dict('cleaned_senti')
    trained_d = read_in_dict('training_sent_dict')
    freq_d = read_in_dict('training_frequency_dict')

    # Read in training file
    print("Creating Training Set")
    training_list = parse_training_set(train_data, sentiWord_d, trained_d, freq_d)
    print("the training file size is {0}".format(len(training_list)))
    min_sentiWord, max_sentiWord, min_learnedWord, max_learnedWord, min_length, max_length = find_max_min(training_list)

    # Read in text file
    print("Training set and sentiment dictionaries created.  Next test predictions")
    test_list = parse_test_set(test_data, sentiWord_d, trained_d, freq_d)
    print("the test file size is {0}".format(len(test_list)))

    ### attributes ###
    print("Predicting Review Sentiments")
    k = 25
    regular_run = False
    multithreading = True*(1-regular_run)
    threads = 8
    multiprocessing = (not multithreading)*(1-regular_run)
    updated_test_list = []
    # Classifying test set using multithreading or multiprocessing or neither
    # k_NN is the classifying function. Arguments of k:_NN are passed as a tuple
    # while (threads < 20):
    t = dt.now().now()
    print("Beginning classification at\n{0}\n".format(dt.now().now()))
    if ( multithreading ):
        print("multithreading")
        pool = ThreadPool(processes=threads)
        async_result = pool.apply_async(k_NN, (k, training_list, test_list))
        updated_test_list = async_result.get()
    elif( multiprocessing ):
        print("multiprocessing")
        pool = Pool(processes=cpu_count())
        updated_test_list = k_NN(k, training_list, test_list)
        async_result = pool.apply_async(k_NN, (k, training_list, test_list))
        updated_test_list = async_result.get()
    else:
        print("processing")
        updated_test_list = k_NN(k, training_list, test_list)

    print(dt.now().now())

    # print results as csv for upload
    print("Printing test output files\n")
    print("The size of the update test set {0}, size of test set: {1}".format(len(updated_test_list),
                                                                              len(test_list)))
    print_results_to_csv(updated_test_list)
    t2 = dt.now().now()
    t1 = "{0}.{1}".format(t.minute, t.second)
    t2 = "{0}.{1}".format(t2.minute, t2.second)
    time_delta = float(t2) - float(t2)

    with open('time_run.txt', 'a') as txt:
        txt.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(k, time_delta, t1, t2,"multithreading"))
        # threads += 1

        # def plot(test_results):
        #     pass
        #     dists=[review.mapping for review in test_set]
        #     sentiments = [review.sentiment for review in test_set]
        #     plt.plot(dists, sentiments)
        #     plt.show()

if __name__ == '__main__':
   main()
