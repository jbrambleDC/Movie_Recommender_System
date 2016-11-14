from pylab import *
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csr_matrix
from sklearn import preprocessing
from sklearn.metrics import pairwise
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
            user_id = int(row_list[0]) - 1
            movie_id = int(row_list[1]) - 1
            rating = float(row_list[2])
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

def lists_to_csr(user_list, movie_list, rating_list, verbose):
    if verbose:
        print("in jagged to csr method")
    max_movie_index = max(movie_list) + 1
    max_user_index = user_list[len(user_list) - 1] + 1
    if verbose:
        print("maxcolumn: {}".format(max_movie_index))
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
def write_csr_to_disk(csr, fn, full_run, verbose):
    t_row_csr = truncate_csr(csr,True,verbose)   #deletes empty rows
    t_col_csr = truncate_csr(csr, False, verbose)    # deletes empty cols
    t_csr = truncate_csr(t_row_csr, False,verbose)  # deleted cols and rows
    if verbose:
        print('csr files to disk'.format(fn))
    dir = 'data/'
    if not full_run:
        dir = 'short/'
    fn = dir + fn
    fn_r = dir + 'truncated_row_csr'
    fn_c = dir + 'truncated_col_csr'
    fn_t = dir +  'truncated_csr'
    np.savez(fn, data=csr.data, indices=csr.indices, indptr=csr.indptr, shape=csr.shape)
    np.savez(fn_r, data=t_row_csr.data, indices=t_row_csr.indices, indptr=t_row_csr.indptr, shape=t_row_csr.shape)
    np.savez(fn_r, data=t_col_csr.data, indices=t_col_csr.indices, indptr=t_col_csr.indptr, shape=t_col_csr.shape)
    np.savez(fn_r, data=t_row_csr.data, indices=t_csr.indices, indptr=t_csr.indptr, shape=t_csr.shape)

def truncate_csr(csr, row=True, verbose=False):
    if verbose:
        print("Truncating csr")
    if row:
        truncated_csr = csr[csr.getnnz(1) > 0]
    else:
        truncated_csr = csr[:, csr.getnnz(0) > 0]

    if verbose:
        print("Truncated CSR shape: {}".format(csr.shape))

    return truncated_csr

def write_list_to_disk(dense_bag_of_movies, fn, verbose):
    if verbose:
        print("writing articles to list {}".format(fn))
    with open(fn, 'w') as csv:
        for row in dense_bag_of_movies:
            csv.write("{0}\n".format(row))

# def cosine_similarity(x, y):
#     sim = metrics.pairwise.cosine_similarity(X,)

# def create_content_similarity_matrix(csr, similarity_type='cosine', verbose):
#     if verbose:
#         print("converting sparse index matrix to articles")
#     if similarity_type == 'cosine':
#         sim_matrix = pairwise.cosine_similarity(np.transpose(csr)) #makes a
#     return sim_matrix

# def create_user_similarity_matrix(csr, similarity_type='cosine', verbose):
#     if verbose:
#         print("converting sparse index matrix to articles")
#     if similarity_type == 'cosine':
#         sim_matrix = pairwise.cosine_similarity(csr) #makes a
#     return

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
        train_fn = 'rating_csr'
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
        train_fn = short_dir + 'rating_csr'
    return rating_fn, movie_actors_fn, movie_directors_fn, movie_genres_fn, movie_tags_fn, user_tags_fn, tags_fn, test_fn, train_fn

def write_means(csr, verbose):
    csr_row_mean = csr.sum(axis=1)
    csr_col_mean = csr.mean(axis=0)
    write_list_to_disk(csr_col_mean, 'column_means.txt', verbose)
    write_list_to_disk(csr_row_mean, 'row_means.txt', verbose)

def main():
    verbose = True
    full_run = True
    frequency_write = True
    if verbose:
        print('starting file')

    rating_fn, movie_actors_fn, movie_directors_fn, movie_genres_fn, movie_tags_fn, user_tags_fn, tags_fn, test_fn , output_fn = \
        get_fns(full_run)
    # convert data to lists
    if verbose:
        print("convert data to lists")
    user_list, movie_list, rating_list, movie_freqs = file_to_list(rating_fn, verbose)

    # convert lists to csrs
    if verbose:
        print("converting index and frequency lists to csr and article lists")

    rating_csr = lists_to_csr(user_list, movie_list, rating_list, verbose)
    # i can use this to create other files
    # if create_articles_file:
    #     articles = create_articles(feature_list, bag_of_indexes_csr, verbose)

    # write cleaned data to file
    if verbose:
        print("writing processed information to file")
        # print(rating_csr)
    write_csr_to_disk(rating_csr, output_fn, full_run, verbose)
    if frequency_write:
        write_list_to_disk(movie_freqs, 'movie_frequencies.txt', verbose)

    if verbose:
        print("------------finished-------------")


if __name__ == '__main__':
    main()
