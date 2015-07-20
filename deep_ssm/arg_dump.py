import json
import sys

if __name__ == '__main__':
	file_name = sys.argv[1]
	relations = [json.loads(x, encoding='utf8') for x in open(file_name)]
	for relation in relations:
		print '%s\t%s' % (relation['Arg1']['RawText'].encode('utf8'), relation['Arg2']['RawText'].encode('utf8'))
	for relation in relations:
		print '%s\t%s' % (relation['Arg2']['RawText'].encode('utf8'), relation['Arg1']['RawText'].encode('utf8'))
