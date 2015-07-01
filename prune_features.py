"""Prune features that are fewer than the specified number"""
from collections import Counter
import argparse


def get_feature_counter(file_name):
	"""Count each feature and put the counts in a Counter
	
	Assume this format
	
	[name]\t[label]\t[feature1] [feature2] ...
	"""
	counter = Counter()
	lines = open(file_name).readlines()
	for line in lines:
		name, label, features = line.strip().split('\t')
		for feature in features.split(' '):
			counter[feature] += 1
	return counter

def rewrite_training_file(file_name, counter, cutoff):
	"""Overwrite the file such that the features are pruned based on the cutoff"""
	write_training_file(file_name, file_name, counter, cutoff)

def write_training_file(file_name, new_file_name, counter, cutoff):
	"""Write the file such that the features are pruned based on the cutoff

	It slows down a bit because we have to re-read the file instead using 
	whatever is already in the memory.
	"""
	with open(file_name) as f:
		lines = f.readlines()
	new_training_file = open(new_file_name, 'w')
	for line in lines:
		name, label, features = line.strip().split('\t')
		features = [x for x in features.split(' ') if counter[x] > cutoff]
		if len(features) == 0: 
			features = ['NO_FEATURE']
		new_training_file.write('%s\t%s\t%s\n' % (name, label, ' '.join(features)))


def prune_features(file_name, cutoff, new_file_name=None):
	counter = get_feature_counter(file_name)
	num_features = len(counter)
	num_reduced_features = len([x for x in counter if counter[x] > cutoff])
	print 'From %s features reduced to %s features' % (num_features, num_reduced_features)
	if new_file_name:
		write_training_file(file_name, new_file_name, counter, cutoff)
	else:
		rewrite_training_file(file_name, counter, cutoff)


if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('file_name', help='the file name of the training set csv that needs pruning', type=str)
	argparser.add_argument('cutoff', help='the cutoff count', default=20, type=int)
	args = argparser.parse_args()
	prune_features(args.file_name, args.cutoff)


