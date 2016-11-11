from pylab import *
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from sklearn import preprocessing
import itertools


max_movies = 65133 # based on actors directors and genres file
def file_to_list(file, verbose=False):
    if verbose:
        print("converting training data to list")
    file = 'data/' + file
    max_movie_count = 0
    movie_list = []
    user_list = []
    rating_list = []
    with open(file, 'r') as csv:
        movie_num = 0
        movie_freqs = [0] * max_movies
        csv.readline()  # skip header
        for row in csv:
            # each row is are movie indexes paired with their counts, all separated by spaces
            row_list = row.split(' ')
            print(row_list)
            user_id = int(row_list[0]) - 1
            movie_id = int(row_list[1]) - 1
            rating = float(row_list[2])
            print(rating)
            if movie_id > max_movie_count:
                max_movie_count = movie_id
            movie_freqs[movie_id] += 1

            # each appended item is a article with its movie (reference indexes) and respective counts
            # creates jagged list with varying length reviews
            user_list.append(user_id)
            movie_list.append(movie_id)
            rating_list.append(rating)
    if verbose:
        print("Max movie counted: {}".format(max_movie_count))

    return user_list, movie_list, rating_list, movie_freqs

def lists_to_csc(user_list, movie_list, rating_list, verbose):
    if verbose:
        print("in jagged to csc method")
    max_movie_index = max(movie_list) + 1
    max_user_index = user_list[len(user_list) - 1] + 1
    if verbose:
        print("maxcolumn: {}".format(max_movie_index))
        print(" {} \n {} \n {}".format(user_list, movie_list, rating_list))
    coo = create_coo(user_list, movie_list, rating_list, max_user_index, max_movie_index, verbose)
    return csr_matrix(coo)

# helper method creates sparse coordinate format matrix for fast construction
def create_coo(rows, cols, values, num_rows, num_cols, verbose):
    if verbose:
        print('creating sparse csr of movies')
        print("rows: {0} \t cols: {1}".format(num_rows, num_cols))
    # in create csr
    sparse_coo = coo_matrix((values, (rows, cols)), shape=(num_rows, num_cols), dtype=np.float32)
    return sparse_coo

# t = training data, x = test data
def write_csc_to_disk(csc, fn, verbose):
    if verbose:
        print('csc file {0} to disk'.format(fn))
    fn = 'data/' + fn
    np.savez(fn, data=csc.data, indices=csc.indices, indptr=csc.indptr, shape=csc.shape)

def write_list_to_disk(dense_bag_of_movies, fn, verbose):
    if verbose:
        print("writing articles to list {}".format(fn))
    fn = 'data/' + fn
    with open(fn, 'w') as csv:
        for row in dense_bag_of_movies:
            csv.write("{0}\n".format(row))

def create_articles(movie_list, index_csc, verbose):
    if verbose:
        print("converting sparse index matrix to articles")
    bag_of_indexes = index_csc.toarray()
    bag_of_articles = [''] * bag_of_indexes.shape[0]  # number of rows
    article_i = 0
    for article in bag_of_indexes:
        movies = []  # assuming appending saves space and time instead of instantiating
        i = 0
        for movie_count in article:
            movie = ''
            if (movie_count > 0):
                movies.append(str(word_list[i] + ' ') * int(word_count))  # multiple by word count for repeated words
            i += 1
        bag_of_articles[article_i] = ("".join(movies))  # convert list of words to string append to bag
        article_i += 1
    return bag_of_articles

def get_fns(full_run=True):
    if full_run:
        rating_fn = 'train.dat'
        movie_actors_fn = 'movie_actors.latin1.dat'
        movie_directors_fn = 'movie_directors.latin1.dat'
        movie_genres_fn = 'movie_genres.dat'
        movie_tags_fn = 'movie_tags.dat'
        user_tags_fn = 'user_taggedmovies.dat'
        tags_fn = 'tags.dat'
        test_fn = 'test.dat'
    else:
        short_dir = 'short/'
        rating_fn = short_dir + 'train_short.dat'
        movie_actors_fn = short_dir + 'movie_actors.latin1_short.dat'
        movie_directors_fn = short_dir + 'movie_directors.latin1_short.dat'
        movie_genres_fn = short_dir + 'movie_genres_short.dat'
        movie_tags_fn = short_dir + 'movie_tags_short.dat'
        user_tags_fn = short_dir + 'user_taggedmovies_short.dat'
        tags_fn = short_dir + 'tags_short.dat'
        test_fn = short_dir + 'test_short.dat'
    return rating_fn, movie_actors_fn, movie_directors_fn, movie_genres_fn, movie_tags_fn, user_tags_fn, tags_fn, test_fn

def write_means(csc, verbose):
    csc_row_mean = csc.mean(axis=1)
    csc_col_mean = csc.mean(axis=0)
    write_list_to_disk(csc_col_mean, 'column_means.txt', verbose)
    write_list_to_disk(csc_row_mean, 'row_means.txt', verbose)

def main():
    verbose = True
    full_run = True
    frequency_write = True
    if verbose:
        print('starting file')

    rating_fn, movie_actors_fn, movie_directors_fn, movie_genres_fn, movie_tags_fn, user_tags_fn, tags_fn, test_fn = \
        get_fns(full_run)
    # convert data to lists
    if verbose:
        print("convert data to lists")
    user_list, movie_list, rating_list, movie_freqs = file_to_list(rating_fn, verbose)

    # convert lists to cscs
    if verbose:
        print("converting index and frequency lists to csc and article lists")

    rating_csc = lists_to_csc(user_list, movie_list, rating_list, verbose)
    # i can use this to create other files
    # if create_articles_file:
    #     articles = create_articles(feature_list, bag_of_indexes_csc, verbose)

    # write cleaned data to file
    if verbose:
        print("writing processed information to file")
        # print(rating_csc)
    write_csc_to_disk(rating_csc, 'rating_csc', verbose)
    if frequency_write:
        write_list_to_disk(movie_freqs, 'movie_frequencies.txt', verbose)

    if verbose:
        print("------------finished-------------")


if __name__ == '__main__':
    main()
