import random

import numpy as np
from scipy import sparse
from os.path import join
from os import makedirs

np.seterr(divide='ignore', invalid='ignore')
data_path = join('data', 'ml-1m', 'ratings.dat')
processed_files_dir = 'processed_data'

all_instances_file_name = 'ratings_all'
train_file_name = 'ratings_train'
test_file_name = 'ratings_test'

test_perc = 0.25        # percentage of testing instances

collaborative_neighbours = 100    # number of neighbours used in collaborative filtering
concepts = 80                       # number of concepts/eigen values to consider while performing SVD decomposition
CUR_no_cols = 4 * concepts          # number of columns and rows to select while performing CUR

txt_dir = join(processed_files_dir, 'txts')
sparse_dir = join(processed_files_dir, 'sparse_matrices')

makedirs(txt_dir, exist_ok=True)
makedirs(sparse_dir, exist_ok=True)

def get_txt_path(name):
    return join(txt_dir, f'{name}.txt')

def get_txt_path_by_type(type):
    if type == 'train':
        return get_txt_path(train_file_name)
    elif type == 'test':
        return get_txt_path(test_file_name)
    elif type == 'all':
        return get_txt_path(all_instances_file_name)
    else:
        raise ValueError('Invalid value for type')

def get_sparse_path(name, normalized = False):
    suffix = '_normalized' if normalized else '_original'
    return join(sparse_dir, f'{name}{suffix}.npz')

def get_sparse_path_by_type(type, normalized = False):
    if type == 'train':
        return get_sparse_path(train_file_name, normalized)
    elif type == 'test':
        return get_sparse_path(test_file_name, normalized)
    elif type == 'all':
        return get_sparse_path(all_instances_file_name, normalized)
    else:
        raise ValueError('Invalid value for type')

def split_train_test():

	# read main file
	ratings = []
	no_users, no_movies = -float('inf'), -float('inf')

	for line in open(data_path, 'r'):
		words = line.split('::')
		ratings.append((words[0], words[1], words[2]))
		no_users = max(no_users, int(words[0]))
		no_movies = max(no_movies, int(words[1]))

	print(f'Number of ratings: {len(ratings)}')
	print(f'Number of users: {no_users}, Number of movies: {no_movies}')

	# create train and test files
	random.shuffle(ratings)
	no_test = int(test_perc * (len(ratings) + 1))

	all_path = get_txt_path_by_type('all')
	train_path = get_txt_path_by_type('train')
	test_path = get_txt_path_by_type('test')

	with open(train_path, 'a') as train_file, open(test_path, 'a') as test_file, open(all_path, 'a') as all_file:
		for i, item in enumerate(ratings):
			line = item[0] + ',' + item[1] + ',' + item[2] + '\n'

			all_file.write(line)
			if i <= no_test:
				test_file.write(line)
			else:
				train_file.write(line)

	return no_users, no_movies


def form_sparse_matrix(type, shape):

	filepath = get_txt_path_by_type(type)

	users_list, movies_list, data_list = [], [], []
	for line in open(filepath):
		values = line.split(',')
		users_list.append(int(values[0]))
		movies_list.append(int(values[1]))
		data_list.append(int(values[2]))

	row = np.array(users_list, dtype = np.float32)
	col = np.array(movies_list, dtype = np.float32)
	row, col = row - 1, col - 1

	data = np.array(data_list, dtype = np.float32)
	sparse_matrix = sparse.coo_matrix((data, (row, col)), shape=shape).tocsr()

	# save the sparse matrix
	sparse_filepath = get_sparse_path_by_type(type)
	sparse.save_npz(sparse_filepath, sparse_matrix)
	print(f'Saved {sparse_filepath}')

	return sparse_matrix


def normalize(sparse_matrix, type):

	row_mean = sparse_matrix.sum(1)/(sparse_matrix!=0).sum(1)
	dense_matrix = sparse_matrix.todense()
	r,c = np.where(dense_matrix == 0)
	sparse_matrix -= row_mean
	sparse_matrix[r,c] = 0
	sparse_matrix = sparse.csr_matrix(sparse_matrix)

	# save the normalized matrix
	sparse_filepath = get_sparse_path_by_type(type, normalized=True)
	sparse.save_npz(sparse_filepath, sparse_matrix)
	print(f'Saved {sparse_filepath}')


def main():

	# split train-test
	print('Splitting into train test ...')
	shape = split_train_test()

	# create and save sparse matrices
	print('\nCreating sparse matrices ...')

	# save all sparse matrices
	for type in ['all', 'train', 'test']:
		sparse_matrix = form_sparse_matrix(type, shape)
		normalize(sparse_matrix, type)

if __name__ == '__main__':
	main()
