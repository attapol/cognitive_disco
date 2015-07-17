import json
import re
import sys

from collections import namedtuple


class SemVectorPair(object):

	def __init__(self, src_vec, target_vec):
		#self.src_vec = '[%s]' % re.sub(' ', ',', src_vec)
		self.src_vec = [float(x) for x in src_vec.split(' ')]
		#self.target_vec = '[%s]' % re.sub(' ', ',', target_vec)
		self.target_vec = [float(x) for x in target_vec.split(' ')]

def parse_dssm_output_file(output_file):
	svp_list = []
	while True:
		sim_line = output_file.readline()
		if sim_line == '':
			break
		src_line = output_file.readline().strip().split(': ')[1]
		target_line = output_file.readline().strip().split(': ')[1]
		svp_list.append(SemVectorPair(src_line, target_line))
	return svp_list

if __name__ == '__main__':
	file_name = sys.argv[1]
	relations = [json.loads(x) for x in open(file_name)]

	dssm_file = open('tmp_dssm_vec.out')
	dssm_vectors = parse_dssm_output_file(dssm_file)
	assert(len(dssm_vectors) == 2 * len(relations))

	cdssm_file = open('tmp_cdssm_vec.out')
	cdssm_vectors = parse_dssm_output_file(cdssm_file)
	assert(len(cdssm_vectors) == 2 * len(relations))

	num_relations = len(relations)
	for i, relation in enumerate(relations):
		relation['Arg1']['DSSMSource'] = dssm_vectors[i].src_vec
		relation['Arg2']['DSSMTarget'] = dssm_vectors[i].target_vec
		relation['Arg1']['CDSSMSource'] = cdssm_vectors[i].src_vec
		relation['Arg2']['CDSSMTarget'] = cdssm_vectors[i].target_vec

		relation['Arg1']['DSSMTarget'] = dssm_vectors[i + num_relations].target_vec
		relation['Arg2']['DSSMSource'] = dssm_vectors[i + num_relations].src_vec
		relation['Arg1']['CDSSMTarget'] = cdssm_vectors[i + num_relations].target_vec
		relation['Arg2']['CDSSMSource'] = cdssm_vectors[i + num_relations].src_vec
		print json.dumps(relation, sort_keys=True)


