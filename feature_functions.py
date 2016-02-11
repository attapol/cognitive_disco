"""Feature functions

Each function should take a data_reader.DRelation object as an argument,
and output a list of feature strings. 
If a function reuses some values from the object over and over,
the implementation should move to the methods not in the feature functions.

"""
import json
import os
import re
import random
import codecs
from nltk.tree import Tree

def random_feature(relation):
    return ['RANDOM:%s' % random.random()]

def first_word(relation):
    return [relation.arg_tokens(1)[0]]

def first3(relation):
    feature_vector = []
    arg_tokens = relation.arg_tokens(1)
    arg_tokens.extend(relation.arg_tokens(2))
    for arg_token in arg_tokens:
        feature = 'BOW_%s' % arg_token
        feature = re.sub(':','COLON', feature)
        feature_vector.append(feature)
    return feature_vector[0:3]

def bag_of_words(relation):
    """Bag of words features

    : needs to be replaced with a string because 
    it will mess up Mallet feature vector converter
    """
    feature_vector = []
    arg_tokens = relation.arg_tokens(1)
    arg_tokens.extend(relation.arg_tokens(2))
    for arg_token in arg_tokens:
        feature = 'BOW_%s' % arg_token
        feature = re.sub(':','COLON', feature)
        feature_vector.append(feature)
    return feature_vector

def word_pairs(relation):
    """Word pair features

    Bob is hungry. He wants a burger --> 
        Bob_He, Bob_wants, Bob_a, ... hungry_burger

    : needs to be replaced with a string because 
    it will mess up Mallet feature vector converter
    """
    feature_vector = []
    for arg1_token in relation.arg_tokens(1):
        for arg2_token in relation.arg_tokens(2):
            feature = 'WP_%s_%s' % (arg1_token, arg2_token)
            feature = re.sub(':','COLON', feature)
            feature_vector.append(feature)
    return feature_vector

def _get_average_vp_length(parse_tree, arg_token_indices):
    if len(parse_tree.leaves()) == 0:
        return 0
    start_index = min(arg_token_indices)
    end_index = max(arg_token_indices) + 1
    if end_index - start_index == 1:
        return 0

    tree_position = parse_tree.treeposition_spanning_leaves(start_index, end_index)
    subtree = parse_tree[tree_position]

    agenda = [subtree]
    while len(agenda) > 0:
        current = agenda.pop(0)
        if current.height() > 2:
            if current.node == 'VP':
                return len(current.leaves())
            for child in current:
                agenda.append(child)
    return 0    

def average_vp_length(relation):
    arg1_tree, token_indices1 = relation.arg_tree(1)
    arg2_tree, token_indices2 = relation.arg_tree(2)
    arg1_average_vp_length = _get_average_vp_length(Tree(arg1_tree), token_indices1)
    arg2_average_vp_length = _get_average_vp_length(Tree(arg2_tree), token_indices2)
    if arg1_average_vp_length == 0 or arg2_average_vp_length == 0: 
        return []
    return ['ARG1_VP_LENGTH=%s' % arg1_average_vp_length,
            'ARG2_VP_LENGTH=%s' % arg2_average_vp_length,
            'VP_LENGTH_%s_%s' % (arg1_average_vp_length, arg2_average_vp_length)]

def _has_modality(words):
    for w in words:
        if w.pos == 'MD':
            return 'HAS_MODALITY'
    return 'NO_MODALITY'

def modality(relation):
    arg1_modality = _has_modality(relation.arg_words(1))
    arg2_modality = _has_modality(relation.arg_words(2))
    feature_vector = ['ARG1_%s' % arg1_modality,
            'ARG2_%s' % arg2_modality, 
            'ARG1_%s_ARG2_%s' % (arg1_modality, arg2_modality)]
    return feature_vector

def is_arg1_multiple_sentences(relation):
    arg1_sentence_indices = set([x.sentence_index for x in relation.arg_words(1)])
    if len(arg1_sentence_indices) > 1:
        return ['ARG1_MULTIPLE_SENTENCES'] 
    return []

def first_last_first_3(relation):
    """First Last First 3 features 
    
    first and last of arg1
    first and last of arg2
    first of arg1 and arg2 together
    last of arg1 and arg2 together
    first three of arg1
    first three of arg2
    """
    first_arg1 = relation.arg_tokens(1)[0]
    last_arg1 = relation.arg_tokens(1)[-1]
    first_arg2 = relation.arg_tokens(1)[0]
    last_arg2 = relation.arg_tokens(1)[-1]
    first_3_arg1 = '_'.join(relation.arg_tokens(1)[:3])
    first_3_arg2 = '_'.join(relation.arg_tokens(2)[:3])

    feature_vector = []
    feature_vector.append(first_arg1)
    feature_vector.append(last_arg1)
    feature_vector.append(first_arg2)
    feature_vector.append(last_arg2)
    feature_vector.append('FIRST_FIRST_%s__%s' % (first_arg1, first_arg2))
    feature_vector.append('LAST_LAST_%s__%s' % (last_arg1, last_arg2))
    feature_vector.append(first_3_arg1)
    feature_vector.append(first_3_arg2)
    return [re.sub(':','COLON',x) for x in feature_vector]

def production_singles(relation):
    arg1_tree, token_indices1 = relation.arg_tree(1)
    arg2_tree, token_indices2 = relation.arg_tree(2)
    rule_set1 = _get_production_rules(Tree(arg1_tree), token_indices1)
    rule_set2 = _get_production_rules(Tree(arg2_tree), token_indices2)
    feature_vector = []
    for rule in rule_set1:
        feature_vector.append('A1RULE=%s' % rule)
    for rule in rule_set2:
        feature_vector.append('A2RULE=%s' % rule)
    return feature_vector

def production_pairs(relation):
    arg1_tree, token_indices1 = relation.arg_tree(1)
    arg2_tree, token_indices2 = relation.arg_tree(2)
    rule_set1 = _get_production_rules(Tree(arg1_tree), token_indices1)
    rule_set2 = _get_production_rules(Tree(arg2_tree), token_indices2)
    feature_vector = []
    for rule1 in rule_set1:
        for rule2 in rule_set2:
            feature_vector.append('RULEPAIR=%s_%s' % (rule1, rule2))
    return feature_vector

def production_rules(relation):
    arg1_tree, token_indices1 = relation.arg_tree(1)
    arg2_tree, token_indices2 = relation.arg_tree(2)
    rule_set1 = _get_production_rules(Tree(arg1_tree), token_indices1)
    rule_set2 = _get_production_rules(Tree(arg2_tree), token_indices2)
    
    #if len(rule_set1) == 0 or len(rule_set2) == 0:
    #    return []

    feature_vector = []
    for rule in rule_set1.intersection(rule_set2):
        feature_vector.append('BOTH_ARGS_RULE=%s' % rule)
    for rule in rule_set1:
        feature_vector.append('ARG1RULE=%s' % rule)
    for rule in rule_set2:
        feature_vector.append('ARG2RULE=%s' % rule)
    return feature_vector
    
def _get_production_rules(parse_tree, token_indices):
    """Find all of the production rules from the subtree that spans over the token indices

    Args:
        parse_tree : an nltk tree object that spans over the sentence that the arg is in
        token_indices : the indices where the arg is.

    Returns:
        a set of production rules used over the argument

    """
    if len(parse_tree.leaves()) == 0:
        return set()
    if len(token_indices) == 1:
        tree_position = parse_tree.leaf_treeposition(token_indices[0])
        arg_subtree = parse_tree[tree_position[0:-1]]
    else:
        start_index = min(token_indices)
        end_index = max(token_indices) + 1
        tree_position = parse_tree.treeposition_spanning_leaves(start_index, end_index)
        arg_subtree = parse_tree[tree_position]

    rule_set = set()
    #try:
    for rule in arg_subtree.productions():
        s = rule.__str__()
        #we want to skip all of the unary production rules
        #if "'" not in s and 'ROOT' not in s:
        if 'ROOT' not in s:
            s = s.replace(' -> ', '->')
            s = s.replace(' ','_')
            s = s.replace(':','COLON')
            rule_set.add(s)
    #except:
        #print rule_set
        #pass
    return rule_set

def _vector_based_feature(vector, prefix):
    feature_vector = ['%s%s:%s' % (prefix, i, x) for i, x in enumerate(vector)]
    return feature_vector

def dssm_feature(rplus):
    feature_vector = []
    feature_vector.extend(
            _vector_based_feature(rplus.relation_dict['Arg1']['DSSMTarget'], 'DT1'))
    feature_vector.extend(
            _vector_based_feature(rplus.relation_dict['Arg2']['DSSMTarget'], 'DT2'))
    feature_vector.extend(
            _vector_based_feature(rplus.relation_dict['Arg1']['DSSMSource'], 'DS1'))
    feature_vector.extend(
            _vector_based_feature(rplus.relation_dict['Arg2']['DSSMSource'], 'DS2'))
    return feature_vector

def cdssm_feature(rplus):
    feature_vector = []
    feature_vector.extend(
            _vector_based_feature(rplus.relation_dict['Arg1']['CDSSMTarget'], 'CT1'))
    feature_vector.extend(
            _vector_based_feature(rplus.relation_dict['Arg2']['CDSSMTarget'], 'CT2'))
    feature_vector.extend(
            _vector_based_feature(rplus.relation_dict['Arg1']['CDSSMSource'], 'CS1'))
    feature_vector.extend(
            _vector_based_feature(rplus.relation_dict['Arg2']['CDSSMSource'], 'CS2'))
    return feature_vector

def dependency_rules(relation):
    rule_set1 = set(relation.arg_dtree_rule_list(1))
    rule_set2 = set(relation.arg_dtree_rule_list(2))
    feature_vector = []
    for rule in rule_set1.intersection(rule_set2):
        feature_vector.append('BOTH_ARGS_DRULE=%s' % rule)
    for rule in rule_set1:
        feature_vector.append('ARG1RULE=%s' % rule)
    for rule in rule_set2:
        feature_vector.append('ARG2RULE=%s' % rule)
    return feature_vector


class LexiconBasedFeaturizer(object):
    def __init__(self):
        home = os.path.expanduser('~')
        self.load_inquirer('%s/nlp/lib/lexicon/inquirer/inquirer_merged.json' % home)
        self.load_mpqa('%s/nlp/lib/lexicon/mpqa_subj_05/mpqa_subj_05.json' % home)
        self.load_levin('%s/nlp/lib/lexicon/levin/levin.json' % home)

    def load_inquirer(self, path):
        """Load Inquirer General Tag corpus

        (WORD) --> [tag1, tag2, ...]
        """
        try:
            lexicon_file = open(path)
            self.inquirer_dict = json.loads(lexicon_file.read())
        except:
            print 'fail to load general inquirer corpus'

    def load_mpqa(self, path):
        """Load MPQA dictionary
        
        (WORD) -->  [positive|negative, strong|weak]
        """
        try:
            lexicon_file = open(path)
            self.mpqa_dict = json.loads(lexicon_file.read())
        except:
            print 'fail to load mpqa corpus'

    def load_levin(self, path):
        """Load Levin's verb class dictionary

        (WORD) --> [class1, class2, ...]
        """
        try:
            lexicon_file = open(path)
            self.levin_dict = json.loads(lexicon_file.read())
        except:
            print 'fail to laod levin verb classes'

    def _get_inquirer_tags(self, words):
        tags = []
        for i, w in enumerate(words):
            key = w.word_token.upper()
            if key in self.inquirer_dict:
                tags.extend(self.inquirer_dict[key])
        return tags
    
    def inquirer_tag_feature(self, relation):
        arg1_tags = self._get_inquirer_tags(relation.arg_words(1))
        arg2_tags = self._get_inquirer_tags(relation.arg_words(2))
        feature_vector = []
        if len(arg1_tags) > 0 and len(arg2_tags) > 0:
            for arg1_tag in arg1_tags:
                for arg2_tag in arg2_tags:
                    feature_vector.append('TAGS=%s_%s' % (arg1_tag, arg2_tag))
        for arg1_tag in arg1_tags:
            feature_vector.append('ARG1_TAG=%s' % arg1_tag)
        for arg2_tag in arg1_tags:
            feature_vector.append('ARG2_TAG=%s' % arg1_tag)
        return feature_vector
    
    def _get_mpqa_score(self, words):
        positive_score = 0
        negative_score = 0
        neg_positive_score = 0
        neutral_score = 0
        for i, word in enumerate(words):
            token = word.word_token.upper()
            if token in self.mpqa_dict:
                polarity = self.mpqa_dict[token][0]
                if i != 0 and polarity == 'positive':
                    preceding_token = words[i-1].word_token.upper()
                    if (preceding_token in self.mpqa_dict and self.mpqa_dict[preceding_token] == 'negative'):
                        neg_positive_score += 1
                    else:
                        positive_score += 1
                elif polarity == 'positive':
                    positive_score += 1
                elif polarity =='negative':
                    negative_score += 1
                elif polarity == 'neutral':
                    neutral_score += 1
        return (positive_score, negative_score, neg_positive_score, neutral_score)

    def mpqa_score_feature(self, relation):
        positive_score1, negative_score1, neg_positive_score1, neutral_score1 = \
                self._get_mpqa_score(relation.arg_words(1))
        positive_score2, negative_score2, neg_positive_score2, neutral_score2 = \
                self._get_mpqa_score(relation.arg_words(2))

        feature_vector1 = []
        feature_vector1.append('Arg1MPQAPositive:%s' % positive_score1)
        feature_vector1.append('Arg1MPQANegative:%s' % negative_score1)
        feature_vector1.append('Arg1MPQANegPositive:%s' % neg_positive_score1)

        feature_vector2 = []
        feature_vector2.append('Arg2MPQAPositive:%s' % positive_score2)
        feature_vector2.append('Arg2MPQANegative:%s' % negative_score2)
        feature_vector2.append('Arg2MPQANegPositive:%s' % neg_positive_score2)

        feature_vector = []
        for f1 in feature_vector1:
            for f2 in feature_vector2:
                feature = '%s__%s' % (f1, f2)
                feature_vector.append(feature.replace(':', 'COLON'))
        feature_vector.extend(feature_vector1)
        feature_vector.extend(feature_vector2)
        return feature_vector

    def _get_levin_verb_tags(self, words):
        verbs_tags = []
        for word in words:
            if word.pos[0] == 'V':
                if word.lemma in self.levin_dict:
                    verbs_tags.append(set(self.levin_dict[word.lemma]))
        return verbs_tags

    def levin_verbs(self, relation):
        arg1_levin_verb_tags = self._get_levin_verb_tags(relation.arg_words(1))        
        arg2_levin_verb_tags = self._get_levin_verb_tags(relation.arg_words(2))
        num_verbs_in_common = 0
        for tags1 in arg1_levin_verb_tags:
            for tags2 in arg2_levin_verb_tags:
                if not tags1.isdisjoint(tags2):
                    num_verbs_in_common += 1
        return ['COMMON_LEVIN_VERBS=%s' % num_verbs_in_common]


class BrownClusterFeaturizer(object):
    """Brown Cluster-based featurizer

    We will only load the lexicon once and reuse it for all instances.
    Python goodness allows us to treat function as an object that is 
    still bound to another object

    lf = BrownClusterFeaturizer()
    lf.brown_pairs <--- this is a feature function that is bound to the lexicon

    """
    EN_BROWN =  'brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1.txt'
    ZH_BROWN =  'brown-gigaword-zh-c1000.txt'

    def __init__(self, brown_cluster_file_name=None):
        self.word_to_brown_mapping = {}
        if brown_cluster_file_name == None:
            brown_cluster_file_name = self.EN_BROWN
        self._load_brown_clusters('resources/%s' % brown_cluster_file_name)

    def _load_brown_clusters(self, path):
        try:
            lexicon_file = codecs.open(path, encoding='utf8')
            for line in lexicon_file:
                m = re.match('(\S+)\t(\S+)', line)
                cluster_assn = m.group(1)
                word = m.group(2)
                #cluster_assn, word, _ = line.split('\t')
                self.word_to_brown_mapping[word] = cluster_assn    
        except Exception as e:
            print e
            print 'fail to load brown cluster data'

    def get_cluster_bag(self, tokens):
        bag = set()
        for token in tokens:
            if token in self.word_to_brown_mapping:
                bag.add(self.word_to_brown_mapping[token])
        return bag

    def brown_words(self, relation):
        arg1_brown_words = self.get_cluster_bag(relation.arg_tokens(1))
        arg2_brown_words = self.get_cluster_bag(relation.arg_tokens(2))
        arg1_only = arg1_brown_words - arg2_brown_words
        arg2_only = arg2_brown_words - arg1_brown_words
        both_args = arg1_brown_words.intersection(arg2_brown_words)
        feature_vector = []
        for brown_word in both_args:
            feature_vector.append('BOTH_ARGS_BROWN=%s' % brown_word)
        for brown_word in arg1_only:
            feature_vector.append('ARG1_BROWN=%s' % brown_word)
        for brown_word in arg2_only:
            feature_vector.append('ARG2_BROWN=%s' % brown_word)
        return feature_vector

    def brown_word_pairs(self, relation):
        """Brown cluster pair features
        
        From the shared task, this feature won NLP People's choice award.
        People like using them because they are so easy to implement and
        work decently well.

        """
        feature_vector = []
        arg1_brown_words = self.get_cluster_bag(relation.arg_tokens(1))
        arg2_brown_words = self.get_cluster_bag(relation.arg_tokens(2))
        for arg1_assn in arg1_brown_words:
            for arg2_assn in arg2_brown_words:
                feature = 'BP_%s_%s' % (arg1_assn, arg2_assn)
                feature_vector.append(feature)
        return feature_vector

