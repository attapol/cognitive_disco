
def doc_id_relation_id_nf(relation):
	return '%s_%s' % (relation.doc_id, relation.relation_id)

def doc_id_nf(relation):
	return relation.doc_id
