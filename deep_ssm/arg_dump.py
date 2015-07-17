import json
import sys

if __name__ == '__main__':
	file_name = sys.argv[1]
	relations = [json.loads(x) for x in open(file_name)]
	for relation in relations:
		print '%s\t%s' % (relation['Arg1']['RawText'], relation['Arg2']['RawText'])
	for relation in relations:
		print '%s\t%s' % (relation['Arg2']['RawText'], relation['Arg1']['RawText'])
