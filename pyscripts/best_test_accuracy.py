import argparse
import sys
import json

def print_best_test_accuracy(file_name, keys):
	results = [json.loads(x) for x in open(file_name)]
	best_result = {}
	best_test_accuracy = 0.0
	key = 'test accuracy'
	extra_info = []
	for result in results:
		if key in result and result[key] > best_test_accuracy:
			best_test_accuracy = result[key]
			best_result = result
			extra_info = [result[x] for x in keys if x in result]
	print '%s %s %s' % (file_name, best_test_accuracy, extra_info)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('files', nargs='+')
	parser.add_argument('--keys', nargs='+')
	#file_names = sys.argv[1:]
	args = parser.parse_args()
	file_names = args.files
	keys = args.keys
	for file_name in file_names:
		print_best_test_accuracy(file_name, keys)

