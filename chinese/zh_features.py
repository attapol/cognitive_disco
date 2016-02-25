import cognitive_disco.feature_functions as f
import cognitive_disco.base_label_functions as l
from cognitive_disco.naming_functions import doc_id_relation_id_nf
from cognitive_disco.feature_file_generator import generate_feature_files 

def powerset(seq, i):
    """Returns a powerset"""
    if i == (len(seq) - 1):
        yield seq[i:]
        yield []
    else:
        for item in powerset(seq, i+1):
            yield [seq[i]] + item
            yield item


def powerset_features(dir_list, label_f, brown_f, naming_f):
    features = [f.word_pairs, f.production_rules, f.dependency_rules, brown_f.brown_word_pairs]
    feature_names = ['wp', 'pr', 'dr', 'bp']
    for feature_set, feature_set_name in zip(powerset(features, 0), powerset(feature_names,0)):
        if len(feature_set) > 1:
            print feature_set_name
            name = '_'.join(feature_set_name)
            generate_feature_files(dir_list, feature_set, [label_f], naming_f, name)

def main():
    dir_list = [
        'conll16st-zh-01-08-2016-train',
        'conll16st-zh-01-08-2016-dev',
        'conll16st-zh-01-08-2016-test',
        ]
    nf = doc_id_relation_id_nf
    lf = l.OriginalLabel()

    bf1000 = f.BrownClusterFeaturizer('/home/j/llc/tet/nlp/lib/lexicon/brown_clusters/gigaword-zh-c1000.txt')
    bf3200 = f.BrownClusterFeaturizer('/home/j/llc/tet/nlp/lib/lexicon/brown_clusters/gigaword-zh-c3200.txt')
    generate_feature_files(dir_list, [bf1000.brown_word_pairs], [lf], nf, 'brown1000c_word_pairs')
    generate_feature_files(dir_list, [bf3200.brown_word_pairs], [lf], nf, 'brown3200c_word_pairs')

    #generate_feature_files(dir_list, [bf.brown_word_pairs], [original_label], nf, 'brown_word_pairs')

    #bf = f.BrownClusterFeaturizer(f.BrownClusterFeaturizer.ZH_BROWN)
    #powerset_features(dir_list, lf, bf, nf)
    #generate_feature_files(dir_list, [f.word_pairs], [original_label], nf, 'word_pairs')
    #generate_feature_files(dir_list, [f.production_rules], [original_label], nf, 'production_rules')
    #generate_feature_files(dir_list, [f.dependency_rules], [original_label], nf, 'dependency_rules')
    #generate_feature_files(dir_list, [bf.brown_word_pairs], [original_label], nf, 'brown_word_pairs')

if __name__ == '__main__':
    main()

