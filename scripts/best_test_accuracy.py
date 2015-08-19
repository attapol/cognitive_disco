import sys
import json

def print_best_test_accuracy(file_name):
	results = [json.loads(x) for x in open(file_name)]
	best_result = {}
	best_test_accuracy = 0.0
	key = 'test accuracy'
	for result in results:
		if key in result and result[key] > best_test_accuracy:
			best_test_accuracy = result[key]
			best_result = result
	print '%s %s' % (file_name, best_test_accuracy)

if __name__ == '__main__':
	file_names = sys.argv[1:]
	for file_name in file_names:
		print_best_test_accuracy(file_name)

