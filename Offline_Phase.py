import numpy as np
from scipy import sparse
import pandas as pd
from datetime import datetime as dt


def get_time():
    time = dt.now()
    hour, minute = str(time.hour), str(time.minute)
    if (len(minute) == 1):
        minute = '0' + minute
    if (len(hour) == 1):
        hour = '0' + hour
    time = hour + minute
    return time

def read_in_truncated_csr(verbose):
    if verbose:
        print("Getting Truncated CSR {}".format(get_time()))
    dir = 'data/processed/'
    fn = dir + 'trunc_rating_csr.npz'
    npy_file = np.load(fn)
    csr = sparse.csr_matrix((npy_file['data'], npy_file['indices'], npy_file['indptr']), shape=npy_file['shape'])
    return csr

def csr_to_dataframe(csr, verbose):
    if verbose:
        print("Converting to dataframe, {}".format(get_time()))
        print("Using todense() because of truncated csr, but inefficient conversion")
    df = pd.DataFrame(csr.todense())
    if verbose:
        print("Converting Zeros to NaNs for calc convenience, {}".format(get_time()))
    df = df.replace(0, np.NaN)  #assuming all zeros are non ratings
    return df

def find_means(df, verbose):
    if verbose:
        print("Finding DF means, {}".format(get_time()))
    row_means = df.mean(axis=1)
    col_means = df.mean(axis=0)
    return row_means, col_means

def mean_center(df, means, ax, verbose):
    if ax == 0:
        if verbose:
            print("Subtracting row means, {}".format(get_time()))
        centered_df = df.sub(means, axis=ax)
    elif ax == 1:
        if verbose:
            print("Subtracting col means, {}".format(get_time()))
        centered_df = df.sub(means, axis=ax)
    return centered_df

def pearson_similarity(df, rows, verbose):
    # pairwise correlation of columns excluding NA
    # assumption  that data is missing randomly
    # can also be spearman which ranks things similar to pearson but not necessarily linear
    if verbose:
        print("Pearson Similarity run {}".format(get_time()))
    if rows:
        similarity_df = df.T.corr(method='pearson', min_periods=1)
    else:
        similarity_df = df.corr(method='pearson', min_periods=1)
    return similarity_df

def find_common_nonzero_count(df, verbose):
    nulls = df.notnull()
    write_counts_to_disk = True
    common_element_list = np.zeros([len(nulls), len(nulls)])
    for i in range(len(nulls)):
        for j in range(len(nulls)):
            common_element_list[i, j] = sum(nulls.loc[i] & nulls.loc[j])

    if(write_counts_to_disk):
        write_to_dsk(common_element_list, 'common_element_list.csv', verbose)

    return common_element_ls

def discounted_similarity(df,rows, verbose):

    pass

def write_df_to_disk(df, fn, verbose):
    if verbose:
        print("writing file to disk: {}".format(fn))
    fn = 'data/processed/' + fn
    df.to_csv(path_or_buf=fn)

def write_to_disk(ls, fn, verbose):
    if verbose:
        print("writing to disk, {}".format(get_time()))
    test_output = 'data/processed/' + fn
    with open(test_output, 'w') as results:
        for y in predictions:
            results.write('{0}\n'.format(y))
    test_output = 'test_output/' + "test_results_with_probabilities" + hour + minute + '.csv'
    df.to_csv(path=test_output)

def main():
    verbose = True
    if verbose:
        print('')
    csr = read_in_truncated_csr(verbose)
    df = csr_to_dataframe(csr, verbose)
    row_means, col_means = find_means(df, verbose)
    row_centered_df = mean_center(df,row_means,0,verbose)
    col_centered_df = mean_center(df,col_means,1,verbose)
    pearson_df = pearson_similarity(df, True, verbose)
    write_df_to_disk(pearson_df, 'user_pearson_similarity_1period',verbose)
    print('---------------FINISHED-----------------')
    print('time is {}'.format(get_time()))


if __name__ == '__main__':
    main()


