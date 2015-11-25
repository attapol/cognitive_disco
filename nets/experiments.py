import json
import sys
import timeit

import numpy as np
import theano.sparse
import theano.tensor as T

from stlstm_experiments import net_experiment_stlstm
from tlstm_experiments import net_experiment_tlstm
from lstm_experiments import net_experiment_lstm, net_experiment_tree_lstm
from mixture_experiments import net_mixture_experiment1, net_mixture_experiment2

from cognitive_disco.data_reader import extract_implicit_relations
from cognitive_disco.nets.bilinear_layer import \
        BilinearLayer, LinearLayer, GlueLayer, \
        MJMModel, MixtureOfExperts

from cognitive_disco.nets.learning import AdagradTrainer, DataTriplet
import cognitive_disco.nets.util as util
import cognitive_disco.dense_feature_functions as df
import cognitive_disco.feature_functions as f
import cognitive_disco.base_label_functions as l


def net_experiment0_0(dir_list, args):
    """This setup should be deprecated"""
    brown_dict = util.BrownDictionary()
    relation_list_list = [util.convert_level2_labels(extract_implicit_relations(dir))
        for dir in dir_list]
    data_triplet, alphabet = brown_dict.get_brown_matrices_data(relation_list_list, False)
    num_features = data_triplet[0][0].shape[1]
    num_outputs = len(alphabet)

    rng = np.random.RandomState(12)
    blm = BilinearLayer(rng, num_features, num_features, num_outputs, activation_fn=None)
    trainer = AdagradTrainer(blm, blm.hinge_loss, 0.01, 0.01)
    trainer.train_minibatch(50, 20, data_triplet[0], data_triplet[1], data_triplet[2])

def net_experiment0_1(dir_list, args):
    """Use Brown word and Brown pair only"""
    lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]

    bf = f.BrownClusterFeaturizer()
    ff_list = [bf.brown_words, bf.brown_word_pairs]
    #ff_list = [f.modality]
    sfeature_matrices, alphabet = util.sparse_featurize(relation_list_list, ff_list)
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
    rng = np.random.RandomState(12)

    lm = LinearLayer(rng, [sfeature_matrices[0].shape[1]], len(label_alphabet), 
            use_sparse=True, Y=T.lvector(), activation_fn=T.nnet.softmax)
    trainer = AdagradTrainer(lm, lm.crossentropy, 0.01, 0.01)
    start_time = timeit.default_timer()
    print trainer.train_minibatch(50, 20, [sfeature_matrices[0], label_vectors[0]],
        [sfeature_matrices[1], label_vectors[1]],
        [sfeature_matrices[2], label_vectors[2]])
    end_time = timeit.default_timer()
    print end_time - start_time 

def net_experiment0_2(dir_list, args):
    """Use feature selection"""
    lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]

    ff_list = [f.production_rules]
    sfeature_matrices, alphabet = util.sparse_featurize(relation_list_list, ff_list)
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
    mi = util.compute_mi(sfeature_matrices[0], label_vectors[0])
    sfeature_matrices = util.prune_feature_matrices(sfeature_matrices, mi, 500)

    rng = np.random.RandomState(12)
    lm = LinearLayer(rng, [sfeature_matrices[0].shape[1]], len(label_alphabet), 
            use_sparse=True, Y=T.lvector(), activation_fn=T.nnet.softmax)
    trainer = AdagradTrainer(lm, lm.crossentropy, 0.01, 0.01)
    start_time = timeit.default_timer()
    trainer.train_minibatch(50, 20, [sfeature_matrices[0], label_vectors[0]],
        [sfeature_matrices[1], label_vectors[1]],
        [sfeature_matrices[2], label_vectors[2]])
    end_time = timeit.default_timer()
    print end_time - start_time 

def net_experiment0_3(dir_list, args):
    """ Use all surface features 
    """
    lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]
    bf = f.BrownClusterFeaturizer()
    plf = f.LexiconBasedFeaturizer()
    ff_list = [
            f.is_arg1_multiple_sentences, 
            f.first_last_first_3, f.average_vp_length, f.modality, 
            plf.inquirer_tag_feature, 
            plf.mpqa_score_feature, plf.levin_verbs, 
            f.production_rules, bf.brown_words, 
            bf.brown_word_pairs
            ]
    sfeature_matrices, alphabet = util.sparse_featurize(relation_list_list, ff_list)
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
    rng = np.random.RandomState(12)
    lm = LinearLayer(rng, [sfeature_matrices[0].shape[1]], len(label_alphabet), 
            use_sparse=True, Y=T.lvector(), activation_fn=T.nnet.softmax)
    trainer = AdagradTrainer(lm, lm.crossentropy, 0.01, 0.01)
    start_time = timeit.default_timer()
    print trainer.train_minibatch(50, 20, [sfeature_matrices[0], label_vectors[0]],
        [sfeature_matrices[1], label_vectors[1]],
        [sfeature_matrices[2], label_vectors[2]])
    end_time = timeit.default_timer()
    print end_time - start_time 

# net_experiment1_x series
# Investigate the efficacy of bilinearity and feature abstraction through hidden layers
#


def _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list, use_hinge_loss=False):
    json_file = set_logger(experiment_name)
    lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]
    sfeature_matrices, alphabet = util.sparse_featurize(relation_list_list, ff_list)
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
    for rep in xrange(15):
        random_seed = rep
        minibatch_size = 50
        n_epochs = 20
        learning_rate = 0.01
        lr_smoother = 0.01

        rng = np.random.RandomState(random_seed)
        if use_hinge_loss:
            lm = LinearLayer(rng, [sfeature_matrices[0].shape[1]], len(label_alphabet), 
                    use_sparse=True, Y=T.lvector(), activation_fn=None)
            trainer = AdagradTrainer(lm, lm.hinge_loss, learning_rate, lr_smoother)
        else:
            lm = LinearLayer(rng, [sfeature_matrices[0].shape[1]], len(label_alphabet), 
                    use_sparse=True, Y=T.lvector(), activation_fn=T.nnet.softmax)
            trainer = AdagradTrainer(lm, lm.crossentropy, learning_rate, lr_smoother)

        start_time = timeit.default_timer()
        best_iter, best_dev_acc, best_test_acc = \
                trainer.train_minibatch(minibatch_size, n_epochs, 
                    [sfeature_matrices[0], label_vectors[0]],
                    [sfeature_matrices[1], label_vectors[1]],
                    [sfeature_matrices[2], label_vectors[2]])
        end_time = timeit.default_timer()
        print end_time - start_time 
        print best_iter, best_dev_acc, best_test_acc
        result_dict = {
                'test accuracy': best_test_acc,
                'best dev accuracy': best_dev_acc,
                'best iter': best_iter,
                'random seed': random_seed,
                'minibatch size': minibatch_size,
                'learning rate': learning_rate,
                'lr smoother': lr_smoother,
                'experiment name': experiment_name,
                'cost function': 'hinge' if use_hinge_loss else 'crossentropy',
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))


def net_experiment1_brown_l(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    bf = f.BrownClusterFeaturizer()
    ff_list = [bf.brown_words]
    _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list)

def net_experiment1_brown_bl(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    bf = f.BrownClusterFeaturizer()
    ff_list = [bf.brown_word_pairs]
    _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list)

def net_experiment1_brown_l_bl(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    bf = f.BrownClusterFeaturizer()
    ff_list = [bf.brown_words, bf.brown_word_pairs]
    _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list)

def net_experiment1_production_l(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    ff_list = [f.production_singles]
    _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list)

def net_experiment1_production_bl(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    ff_list = [f.production_pairs]
    _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list)

def net_experiment1_production_l_bl(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    ff_list = [f.production_singles, f.production_pairs]
    _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list)

def net_experiment1_brown_l2(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    bf = f.BrownClusterFeaturizer()
    ff_list = [bf.brown_words]
    _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list, use_hinge_loss=True)

def net_experiment1_brown_bl2(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    bf = f.BrownClusterFeaturizer()
    ff_list = [bf.brown_word_pairs]
    _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list, use_hinge_loss=True)

def net_experiment1_brown_l_bl2(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    bf = f.BrownClusterFeaturizer()
    ff_list = [bf.brown_words, bf.brown_word_pairs]
    _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list, use_hinge_loss=True)

def net_experiment1_production_l2(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    ff_list = [f.production_singles]
    _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list, use_hinge_loss=True)

def net_experiment1_production_bl2(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    ff_list = [f.production_pairs]
    _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list, use_hinge_loss=True)

def net_experiment1_production_l_bl2(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    ff_list = [f.production_singles, f.production_pairs]
    _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list, use_hinge_loss=True)

def net_experiment1_brown_production_l_bl(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    bf = f.BrownClusterFeaturizer()
    ff_list = [bf.brown_words, bf.brown_word_pairs, f.production_singles, f.production_pairs]
    _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list, use_hinge_loss=False)

def net_experiment1_brown_production_l_bl2(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    bf = f.BrownClusterFeaturizer()
    ff_list = [bf.brown_words, bf.brown_word_pairs, f.production_singles, f.production_pairs]
    _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list, use_hinge_loss=True)


def net_experiment1_word2vec_l(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    json_file = set_logger(experiment_name)
    lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]

    ef = df.EmbeddingFeaturizer()
    data_list = [ef.additive_args(relation_list) for relation_list in relation_list_list]
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
    data_triplet = DataTriplet(data_list, [[x] for x in label_vectors])
    
    for rep in xrange(30):
        random_seed = rep
        minibatch_size = 50
        n_epochs = 20
        learning_rate = 0.01
        lr_smoother = 0.01

        rng = np.random.RandomState(random_seed)
        lm = LinearLayer(rng, data_triplet.input_dimensions(), len(label_alphabet), 
                use_sparse=False, Y=T.lvector(), activation_fn=T.nnet.softmax)
        trainer = AdagradTrainer(lm, lm.crossentropy, learning_rate, lr_smoother)
        start_time = timeit.default_timer()
        best_iter, best_dev_acc, best_test_acc = \
                trainer.train_minibatch_triplet(minibatch_size, n_epochs, data_triplet)
        end_time = timeit.default_timer()
        print end_time - start_time 
        print best_iter, best_dev_acc, best_test_acc
        result_dict = {
                'test accuracy': best_test_acc,
                'best dev accuracy': best_dev_acc,
                'best iter': best_iter,
                'random seed': random_seed,
                'minibatch size': minibatch_size,
                'learning rate': learning_rate,
                'lr smoother': lr_smoother,
                'experiment name': experiment_name,
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))

def net_experiment1_word2vec_bl(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    json_file = set_logger(experiment_name)
    lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]

    ef = df.EmbeddingFeaturizer()
    data_list = [ef.additive_args(relation_list) for relation_list in relation_list_list]
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
    data_triplet = DataTriplet(data_list, [[x] for x in label_vectors])
    for rep in xrange(30):
        random_seed = rep
        minibatch_size = 50
        n_epochs = 20
        learning_rate = 0.01
        lr_smoother = 0.01

        rng = np.random.RandomState(random_seed)

        blm = BilinearLayer(rng, data_triplet.input_dimensions()[0],
            data_triplet.input_dimensions()[1], len(label_alphabet),
            Y=T.lvector(), activation_fn=T.nnet.softmax)

        trainer = AdagradTrainer(blm, blm.crossentropy, learning_rate, lr_smoother)
        start_time = timeit.default_timer()
        best_iter, best_dev_acc, best_test_acc = \
                trainer.train_minibatch_triplet(minibatch_size, n_epochs, data_triplet)
        end_time = timeit.default_timer()
        print end_time - start_time 
        print best_iter, best_dev_acc, best_test_acc
        result_dict = {
                'test accuracy': best_test_acc,
                'best dev accuracy': best_dev_acc,
                'best iter': best_iter,
                'random seed': random_seed,
                'minibatch size': minibatch_size,
                'learning rate': learning_rate,
                'lr smoother': lr_smoother,
                'experiment name': experiment_name,
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))

def net_experiment1_word2vec_l_bl(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    json_file = set_logger(experiment_name)
    lf = l.SecondLevelLabel()
    if len(args) > 0 and args[0] == 'test':
        print 'Using test mode'
        dir_list = ['conll15-st-05-19-15-dev', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
        dict_file = df.EmbeddingFeaturizer.TEST_WORD_EMBEDDING_FILE
        num_reps = 2
    else:
        dict_file = df.EmbeddingFeaturizer.WORD_EMBEDDING_FILE
        num_reps = 30

    relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]
    ef = df.EmbeddingFeaturizer(dict_file)
    data_list = [ef.additive_args(relation_list) for relation_list in relation_list_list]
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
    data_triplet = DataTriplet(data_list, [[x] for x in label_vectors])
    for rep in xrange(num_reps):
        random_seed = rep
        minibatch_size = 50
        n_epochs = 20
        learning_rate = 0.01
        lr_smoother = 0.01

        rng = np.random.RandomState(random_seed)

        X_list = [T.matrix(), T.matrix()]
        lm = LinearLayer(rng, 
                n_in_list=data_triplet.input_dimensions(), 
                n_out=len(label_alphabet), 
                use_sparse=False, 
                X_list=X_list, 
                activation_fn=None)
        blm = BilinearLayer(rng, 
                n_in1=data_triplet.input_dimensions()[0],
                n_in2=data_triplet.input_dimensions()[1], 
                n_out=len(label_alphabet),
                X1=X_list[0], 
                X2=X_list[1],
                activation_fn=None)
        layer = GlueLayer(layer_list=[lm, blm], X_list=X_list, Y=T.lvector(),
                activation_fn=T.nnet.softmax)

        trainer = AdagradTrainer(layer, layer.crossentropy, learning_rate, lr_smoother)
        start_time = timeit.default_timer()
        best_iter, best_dev_acc, best_test_acc = \
                trainer.train_minibatch_triplet(minibatch_size, n_epochs, data_triplet)
        end_time = timeit.default_timer()
        print end_time - start_time 
        print best_iter, best_dev_acc, best_test_acc
        result_dict = {
                'test accuracy': best_test_acc,
                'best dev accuracy': best_dev_acc,
                'best iter': best_iter,
                'random seed': random_seed,
                'minibatch size': minibatch_size,
                'learning rate': learning_rate,
                'lr smoother': lr_smoother,
                'experiment name': experiment_name,
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))

def net_experiment1_word2vec_l2(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    json_file = set_logger(experiment_name)
    lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]

    ef = df.EmbeddingFeaturizer()
    data_list = [ef.additive_args(relation_list) for relation_list in relation_list_list]
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
    data_triplet = DataTriplet(data_list, [[x] for x in label_vectors])
    
    for rep in xrange(30):
        random_seed = rep
        minibatch_size = 50
        n_epochs = 20
        learning_rate = 0.01
        lr_smoother = 0.01

        rng = np.random.RandomState(random_seed)
        lm = LinearLayer(rng, data_triplet.input_dimensions(), len(label_alphabet), 
                use_sparse=False, Y=T.lvector(), activation_fn=None)
        trainer = AdagradTrainer(lm, lm.hinge_loss, learning_rate, lr_smoother)
        start_time = timeit.default_timer()
        best_iter, best_dev_acc, best_test_acc = \
                trainer.train_minibatch_triplet(minibatch_size, n_epochs, data_triplet)
        end_time = timeit.default_timer()
        print end_time - start_time 
        print best_iter, best_dev_acc, best_test_acc
        result_dict = {
                'test accuracy': best_test_acc,
                'best dev accuracy': best_dev_acc,
                'best iter': best_iter,
                'random seed': random_seed,
                'minibatch size': minibatch_size,
                'learning rate': learning_rate,
                'lr smoother': lr_smoother,
                'experiment name': experiment_name,
                'cost function': 'hinge'
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))

def net_experiment1_word2vec_bl2(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    json_file = set_logger(experiment_name)
    lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]

    ef = df.EmbeddingFeaturizer()
    data_list = [ef.additive_args(relation_list) for relation_list in relation_list_list]
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
    data_triplet = DataTriplet(data_list, [[x] for x in label_vectors])
    for rep in xrange(30):
        random_seed = rep
        minibatch_size = 50
        n_epochs = 20
        learning_rate = 0.01
        lr_smoother = 0.01

        rng = np.random.RandomState(random_seed)

        blm = BilinearLayer(rng, data_triplet.input_dimensions()[0],
            data_triplet.input_dimensions()[1], len(label_alphabet),
            Y=T.lvector(), activation_fn=None)

        trainer = AdagradTrainer(blm, blm.hinge_loss, learning_rate, lr_smoother)
        start_time = timeit.default_timer()
        best_iter, best_dev_acc, best_test_acc = \
                trainer.train_minibatch_triplet(minibatch_size, n_epochs, data_triplet)
        end_time = timeit.default_timer()
        print end_time - start_time 
        print best_iter, best_dev_acc, best_test_acc
        result_dict = {
                'test accuracy': best_test_acc,
                'best dev accuracy': best_dev_acc,
                'best iter': best_iter,
                'random seed': random_seed,
                'minibatch size': minibatch_size,
                'learning rate': learning_rate,
                'lr smoother': lr_smoother,
                'experiment name': experiment_name,
                'cost function': 'hinge'
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))

def net_experiment1_word2vec_l_bl2(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    json_file = set_logger(experiment_name)
    lf = l.SecondLevelLabel()
    if len(args) > 0 and args[0] == 'test':
        print 'Using test mode'
        dir_list = ['conll15-st-05-19-15-dev', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
        dict_file = df.EmbeddingFeaturizer.TEST_WORD_EMBEDDING_FILE
        num_reps = 2
    else:
        dict_file = df.EmbeddingFeaturizer.WORD_EMBEDDING_FILE
        num_reps = 30

    relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]
    ef = df.EmbeddingFeaturizer(dict_file)
    data_list = [ef.additive_args(relation_list) for relation_list in relation_list_list]
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
    data_triplet = DataTriplet(data_list, [[x] for x in label_vectors])
    for rep in xrange(num_reps):
        random_seed = rep
        minibatch_size = 50
        n_epochs = 20
        learning_rate = 0.01
        lr_smoother = 0.01

        rng = np.random.RandomState(random_seed)

        X_list = [T.matrix(), T.matrix()]
        lm = LinearLayer(rng, 
                n_in_list=data_triplet.input_dimensions(), 
                n_out=len(label_alphabet), 
                use_sparse=False, 
                X_list=X_list, 
                activation_fn=None)
        blm = BilinearLayer(rng, 
                n_in1=data_triplet.input_dimensions()[0],
                n_in2=data_triplet.input_dimensions()[1], 
                n_out=len(label_alphabet),
                X1=X_list[0], 
                X2=X_list[1],
                activation_fn=None)
        layer = GlueLayer(layer_list=[lm, blm], X_list=X_list, Y=T.lvector(),
                activation_fn=None)

        trainer = AdagradTrainer(layer, layer.hinge_loss, learning_rate, lr_smoother)
        start_time = timeit.default_timer()
        best_iter, best_dev_acc, best_test_acc = \
                trainer.train_minibatch_triplet(minibatch_size, n_epochs, data_triplet)
        end_time = timeit.default_timer()
        print end_time - start_time 
        print best_iter, best_dev_acc, best_test_acc
        result_dict = {
                'test accuracy': best_test_acc,
                'best dev accuracy': best_dev_acc,
                'best iter': best_iter,
                'random seed': random_seed,
                'minibatch size': minibatch_size,
                'learning rate': learning_rate,
                'lr smoother': lr_smoother,
                'experiment name': experiment_name,
                'cost function': 'hinge'
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))

def _run_simple_net(experiment_name, dense_ff, num_reps, dir_list, use_linear=True, use_bilinear=True, use_hinge=True):
    json_file = set_logger(experiment_name)
    lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]
    data_list = [dense_ff(relation_list) for relation_list in relation_list_list]
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
    data_triplet = DataTriplet(data_list, [[x] for x in label_vectors])
    for rep in xrange(num_reps):
        random_seed = rep
        minibatch_size = 50
        n_epochs = 20
        learning_rate = 0.01
        lr_smoother = 0.01

        rng = np.random.RandomState(random_seed)

        X_list = [T.matrix(), T.matrix()]
        layer_list = []
        if use_linear:
            lm = LinearLayer(rng, 
                    n_in_list=data_triplet.input_dimensions(), 
                    n_out=len(label_alphabet), 
                    use_sparse=False, 
                    X_list=X_list, 
                    activation_fn=None)
            layer_list.append(lm)
        if use_bilinear:
            blm = BilinearLayer(rng, 
                    n_in1=data_triplet.input_dimensions()[0],
                    n_in2=data_triplet.input_dimensions()[1], 
                    n_out=len(label_alphabet),
                    X1=X_list[0], 
                    X2=X_list[1],
                    activation_fn=None)
            layer_list.append(blm)
        layer = GlueLayer(layer_list=layer_list, X_list=X_list, Y=T.lvector(),
                activation_fn=None if use_hinge else T.nnet.softmax)

        trainer = AdagradTrainer(layer, layer.hinge_loss if use_hinge else layer.crossentropy,
                learning_rate, lr_smoother)
        start_time = timeit.default_timer()
        best_iter, best_dev_acc, best_test_acc = \
                trainer.train_minibatch_triplet(minibatch_size, n_epochs, data_triplet)
        end_time = timeit.default_timer()
        print end_time - start_time 
        print best_iter, best_dev_acc, best_test_acc
        result_dict = {
                'test accuracy': best_test_acc,
                'best dev accuracy': best_dev_acc,
                'best iter': best_iter,
                'random seed': random_seed,
                'minibatch size': minibatch_size,
                'learning rate': learning_rate,
                'lr smoother': lr_smoother,
                'experiment name': experiment_name,
                'cost function': 'hinge' if use_hinge else 'crossentropy',
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))


def net_experiment1_cdssm_l(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    if len(args) > 0 and args[0] == 'test':
        print 'Using test mode'
        dir_list = ['conll15-st-05-19-15-dev', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
        num_reps = 2
    else:
        num_reps = 30
    _run_simple_net(experiment_name+'_hinge', df.cdssm_feature, num_reps, dir_list, 
            use_linear=True, use_bilinear=False, use_hinge=True)
    _run_simple_net(experiment_name+'_xe', df.cdssm_feature, num_reps, dir_list, 
            use_linear=True, use_bilinear=False, use_hinge=False)

def net_experiment1_cdssm_bl(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    num_reps = 30
    _run_simple_net(experiment_name+'_hinge', df.cdssm_feature, num_reps, dir_list, 
            use_linear=False, use_bilinear=True, use_hinge=True)
    _run_simple_net(experiment_name+'_xe', df.cdssm_feature, num_reps, dir_list, 
            use_linear=False, use_bilinear=True, use_hinge=False)

def net_experiment1_cdssm_l_bl(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    num_reps = 30
    _run_simple_net(experiment_name+'_hinge', df.cdssm_feature, num_reps, dir_list, 
            use_linear=True, use_bilinear=True, use_hinge=True)
    _run_simple_net(experiment_name+'_xe', df.cdssm_feature, num_reps, dir_list, 
            use_linear=True, use_bilinear=True, use_hinge=False)

# net_experiment2_x series
# Merciless Joint Model for Discourse (working title)
# We will use dimensions and use neural to decode the dimensions themselves
#
def net_experiment2_0(dir_list, args):
    # All dimensions use the same features 
    experiment_name = sys._getframe().f_code.co_name    
    json_file = set_logger(experiment_name)
    sense_lf = l.SecondLevelLabel()

    relation_list_list = [extract_implicit_relations(dir, sense_lf) for dir in dir_list]

    bf = f.BrownClusterFeaturizer()
    ff_list = [bf.brown_words]
    data_list, alphabet = util.sparse_featurize(relation_list_list, ff_list)
    
    lf_list = l.GenericMapping('mapping3a.json').get_all_label_functions()
    lf_list.append(sense_lf)
    label_vectors_list = [] #sort by features
    label_alphabet_list = []
    for lf in lf_list:
        label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
        label_alphabet_list.append(label_alphabet)
        label_vectors_list.append(label_vectors)
    label_vectors_list = zip(*label_vectors_list) #sort by data split
    data_triplet = DataTriplet([[x] for x in data_list], label_vectors_list)
    for rep in xrange(30):
        random_seed = rep
        minibatch_size = 50
        n_epochs = 20
        learning_rate = 0.01
        lr_smoother = 0.01

        rng = np.random.RandomState(random_seed)
    
        dimension_layers = []
        sense_label_alphabet = label_alphabet_list[-1]
        dimension_alphabet_list = label_alphabet_list[:-1]
        #input_variables = [theano.sparse.csr_matrix() for i in range(data_triplet.num_input_variables())]
        X = theano.sparse.csr_matrix()
        for i, dimension_alphabet in enumerate(dimension_alphabet_list):
            lm = LinearLayer(rng, 
                    n_in_list=data_triplet.input_dimensions(), 
                    n_out=len(dimension_alphabet), 
                    use_sparse=True, 
                    X_list=[X],
                    Y=T.lvector(),
                    activation_fn=T.nnet.softmax)
            dimension_layers.append(lm)
        sense_layer = LinearLayer(rng,
                n_in_list=[len(x) for x in dimension_alphabet_list] + data_triplet.input_dimensions(),
                n_out=len(sense_label_alphabet),
                use_sparse=True,
                X_list=[layer.activation for layer in dimension_layers] + [X],
                Y=T.lvector(),
                activation_fn=T.nnet.softmax)

        layer_list = []
        layer_list.extend(dimension_layers)
        layer_list.append(sense_layer)
        mjm = MJMModel(layer_list, [X])

        trainer = AdagradTrainer(mjm, mjm.crossentropy, learning_rate, lr_smoother)
        start_time = timeit.default_timer()
        best_iter, best_dev_acc, best_test_acc = \
                trainer.train_minibatch_triplet(minibatch_size, n_epochs, data_triplet)
        end_time = timeit.default_timer()
        print end_time - start_time 
        print best_iter, best_dev_acc, best_test_acc
        result_dict = {
                'test accuracy': best_test_acc,
                'best dev accuracy': best_dev_acc,
                'best iter': best_iter,
                'random seed': random_seed,
                'minibatch size': minibatch_size,
                'learning rate': learning_rate,
                'lr smoother': lr_smoother,
                'experiment name': experiment_name,
                'cost function' : 'sum of hinge losses'
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))

# net_experiment3_x series
# Mixture of Experts to show that there is something special
# and the two ways of classification complement each other
#

def _net_experiment3_dnn_helper(experiment_name, json_file,
        num_hidden_layers, num_hidden_units, num_reps, data_triplet, use_hinge, 
        noise_cache=False):
    #theano.config.optimizer = 'None'
    for rep in xrange(num_reps):
        random_seed = rep
        minibatch_size = 50
        n_epochs = 20
        learning_rate = 0.01
        lr_smoother = 0.01
        rng = np.random.RandomState(random_seed)

        X_list = [T.matrix(), T.matrix(), T.matrix()]
        first_layer = LinearLayer(rng, 
            n_in_list=data_triplet.input_dimensions()[0:2], 
            n_out=data_triplet.output_dimensions()[0] if num_hidden_layers == 0 else num_hidden_units,
            use_sparse=False, 
            X_list=X_list[0:2], 
            activation_fn=None if use_hinge else T.nnet.softmax)
        top_layer = first_layer
        for i in range(num_hidden_layers):
            is_top_layer = i == (num_hidden_layers - 1)
            if is_top_layer:
                hidden_layer = LinearLayer(rng,
                    n_in_list=[num_hidden_units], 
                    n_out=data_triplet.output_dimensions()[0], 
                    use_sparse=False, 
                    X_list=[top_layer.activation], 
                    activation_fn=None if use_hinge else T.nnet.softmax)
            else:
                hidden_layer = LinearLayer(rng,
                    n_in_list=[num_hidden_units], 
                    n_out=num_hidden_units, 
                    use_sparse=False, 
                    X_list=[top_layer.activation], 
                    activation_fn=T.tanh)
            hidden_layer.params.extend(top_layer.params)
            top_layer = hidden_layer

        if noise_cache:
            num_cache_hidden_units = 5
            input_cached_layer = LinearLayer(rng,
                    n_in_list=data_triplet.input_dimensions()[2:3],
                    n_out=num_cache_hidden_units,
                    use_sparse=False,
                    X_list=X_list[2:3],
                    activation_fn=T.tanh)
            cached_layer = LinearLayer(rng,
                    n_in_list=[num_cache_hidden_units],
                    n_out=data_triplet.output_dimensions()[0],
                    use_sparse=False,
                    X_list=[input_cached_layer.activation],
                    activation_fn=None if use_hinge else T.nnet.softmax)
            cached_layer.params.extend(input_cached_layer.params)
        else:
            cached_layer = LinearLayer(rng,
                    n_in_list=data_triplet.input_dimensions()[2:3],
                    n_out=data_triplet.output_dimensions()[0],
                    use_sparse=False,
                    X_list=X_list[2:3],
                    activation_fn=None if use_hinge else T.nnet.softmax)


        moe = MixtureOfExperts(rng, 
                n_in_list=data_triplet.input_dimensions(),
                expert_list=[top_layer, cached_layer],
                X_list=X_list,
                Y=T.lvector())

        trainer = AdagradTrainer(moe, moe.hinge_loss if use_hinge else moe.crossentropy, 
                learning_rate, lr_smoother)
        start_time = timeit.default_timer()
        best_iter, best_dev_acc, best_test_acc = \
                trainer.train_minibatch_triplet(minibatch_size, n_epochs, data_triplet)
        end_time = timeit.default_timer()
        print end_time - start_time 
        print best_iter, best_dev_acc, best_test_acc
        result_dict = {
                'test accuracy': best_test_acc,
                'best dev accuracy': best_dev_acc,
                'best iter': best_iter,
                'random seed': random_seed,
                'minibatch size': minibatch_size,
                'learning rate': learning_rate,
                'lr smoother': lr_smoother,
                'experiment name': experiment_name,
                'cost function': 'hinge' if use_hinge else 'crossentropy',
                'num hidden layers': num_hidden_layers,
                'num hidden units': num_hidden_units,
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))


def _net_experiment3_helper(experiment_name, json_file, num_reps, data_triplet, use_linear, use_bilinear, use_hinge):
    for rep in xrange(num_reps):
        random_seed = rep
        minibatch_size = 50
        n_epochs = 20
        learning_rate = 0.01
        lr_smoother = 0.01

        rng = np.random.RandomState(random_seed)

        X_list = [T.matrix(), T.matrix(), T.matrix()]
        layer_list = []
        if use_linear:
            lm = LinearLayer(rng, 
                    n_in_list=data_triplet.input_dimensions()[0:2], 
                    n_out=data_triplet.output_dimensions()[0], 
                    use_sparse=False, 
                    X_list=X_list[0:2], 
                    activation_fn=None)
            layer_list.append(lm)
        if use_bilinear:
            blm = BilinearLayer(rng, 
                    n_in1=data_triplet.input_dimensions()[0],
                    n_in2=data_triplet.input_dimensions()[1], 
                    n_out=data_triplet.output_dimensions()[0],
                    X1=X_list[0], 
                    X2=X_list[1],
                    activation_fn=None)
            layer_list.append(blm)
        semantic_layer = GlueLayer(layer_list=layer_list, X_list=X_list, Y=T.lvector(),
                activation_fn=None if use_hinge else T.nnet.softmax)
        cached_layer = LinearLayer(rng,
                n_in_list=data_triplet.input_dimensions()[2:3],
                n_out=data_triplet.output_dimensions()[0],
                use_sparse=False,
                X_list=X_list[2:3],
                activation_fn=None if use_hinge else T.nnet.softmax)
        moe = MixtureOfExperts(rng, 
                n_in_list=data_triplet.input_dimensions(),
                expert_list=[semantic_layer, cached_layer],
                X_list=X_list,
                Y=T.lvector())
        trainer = AdagradTrainer(moe, moe.hinge_loss if use_hinge else moe.crossentropy, 
                learning_rate, lr_smoother)
        start_time = timeit.default_timer()
        best_iter, best_dev_acc, best_test_acc = \
                trainer.train_minibatch_triplet(minibatch_size, n_epochs, data_triplet)
        end_time = timeit.default_timer()
        print end_time - start_time 
        print best_iter, best_dev_acc, best_test_acc
        result_dict = {
                'test accuracy': best_test_acc,
                'best dev accuracy': best_dev_acc,
                'best iter': best_iter,
                'random seed': random_seed,
                'minibatch size': minibatch_size,
                'learning rate': learning_rate,
                'lr smoother': lr_smoother,
                'experiment name': experiment_name,
                'cost function': 'hinge' if use_hinge else 'crossentropy',
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))
        


def net_experiment3_additive_l(dir_list, args):
    """MOE = Baseline + Additive arg vectors
    """
    experiment_name = sys._getframe().f_code.co_name    
    sense_lf = l.SecondLevelLabel()

    if len(args) > 0 and args[0] == 'test':
        print 'Using test mode'
        dir_list = ['conll15-st-05-19-15-dev', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
        dict_file = df.EmbeddingFeaturizer.TEST_WORD_EMBEDDING_FILE
        num_reps = 2
    else:
        dict_file = df.EmbeddingFeaturizer.WORD_EMBEDDING_FILE
        num_reps = 10
    json_file = set_logger(experiment_name)

    relation_list_list = [extract_implicit_relations(dir, sense_lf) for dir in dir_list]

    word2vec = df.EmbeddingFeaturizer(dict_file)
    data_list = []
    for relation_list in relation_list_list:
        data = []
        data.extend(word2vec.additive_args(relation_list))
        data.extend(df.cached_features(relation_list, 'BaselineClassification'))
        data_list.append(data)
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, sense_lf)
    data_triplet = DataTriplet(data_list, [[x] for x in label_vectors], [label_alphabet])
    _net_experiment3_helper(experiment_name, json_file, num_reps, data_triplet, 
            use_linear=True, use_bilinear=False, use_hinge=True)
    _net_experiment3_helper(experiment_name, json_file, num_reps, data_triplet, 
            use_linear=True, use_bilinear=False, use_hinge=False)

def net_experiment3_additive_l_dnn(dir_list, args):
    """MOE = Baseline + Additive arg vectors
    """
    experiment_name = sys._getframe().f_code.co_name    
    sense_lf = l.SecondLevelLabel()

    if args[0] == 'test':
        print 'Using test mode'
        dir_list = ['conll15-st-05-19-15-dev', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
        dict_file = df.EmbeddingFeaturizer.TEST_WORD_EMBEDDING_FILE
        num_reps = 2
        num_hidden_layers = 1
    else:
        dict_file = df.EmbeddingFeaturizer.WORD_EMBEDDING_FILE
        num_reps = 10
        num_hidden_layers = int(args[0])
    json_file = set_logger(experiment_name+'_%sh' % num_hidden_layers)

    relation_list_list = [extract_implicit_relations(dir, sense_lf) for dir in dir_list]
    num_hidden_unit_list = [50, 200, 300, 600, 800] 

    word2vec = df.EmbeddingFeaturizer(dict_file)
    data_list = []
    for relation_list in relation_list_list:
        data = []
        data.extend(word2vec.additive_args(relation_list))
        data.extend(df.cached_features(relation_list, 'BaselineClassification'))
        data_list.append(data)
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, sense_lf)
    data_triplet = DataTriplet(data_list, [[x] for x in label_vectors], [label_alphabet])

    for num_hidden_units in num_hidden_unit_list:
        _net_experiment3_dnn_helper(experiment_name, json_file, 
                num_hidden_layers, num_hidden_units, num_reps, data_triplet, 
                use_hinge=True)
        _net_experiment3_dnn_helper(experiment_name, json_file, 
                num_hidden_layers, num_hidden_units, num_reps, data_triplet, 
                use_hinge=False)

def net_experiment3_additive_l_dnn_v2(dir_list, args):
    """MOE = Baseline + Additive arg vectors

    Add a hidden layer to the cached layer as well
    """
    experiment_name = sys._getframe().f_code.co_name    
    sense_lf = l.SecondLevelLabel()

    if args[0] == 'test':
        print 'Using test mode'
        dir_list = ['conll15-st-05-19-15-dev', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
        dict_file = df.EmbeddingFeaturizer.TEST_WORD_EMBEDDING_FILE
        num_reps = 2
        num_hidden_layers = 1
    else:
        dict_file = df.EmbeddingFeaturizer.WORD_EMBEDDING_FILE
        num_reps = 10
        num_hidden_layers = int(args[0])
    json_file = set_logger(experiment_name+'_%sh' % num_hidden_layers)

    relation_list_list = [extract_implicit_relations(dir, sense_lf) for dir in dir_list]
    num_hidden_unit_list = [50, 200, 300, 600, 800] 

    word2vec = df.EmbeddingFeaturizer(dict_file)
    data_list = []
    for relation_list in relation_list_list:
        data = []
        data.extend(word2vec.additive_args(relation_list))
        data.extend(df.cached_features(relation_list, 'BaselineClassification'))
        data_list.append(data)
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, sense_lf)
    data_triplet = DataTriplet(data_list, [[x] for x in label_vectors], [label_alphabet])

    for num_hidden_units in num_hidden_unit_list:
        _net_experiment3_dnn_helper(experiment_name, json_file, 
                num_hidden_layers, num_hidden_units, num_reps, data_triplet, 
                use_hinge=True, noise_cache=True)
        _net_experiment3_dnn_helper(experiment_name, json_file, 
                num_hidden_layers, num_hidden_units, num_reps, data_triplet, 
                use_hinge=False, noise_cache=True)



def net_experiment3_additive_l_bl(dir_list, args):
    """MOE = Baseline + Additive arg vectors
    """
    experiment_name = sys._getframe().f_code.co_name    
    json_file = set_logger(experiment_name)
    sense_lf = l.SecondLevelLabel()

    if len(args) > 0 and args[0] == 'test':
        print 'Using test mode'
        dir_list = ['conll15-st-05-19-15-dev', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
        dict_file = df.EmbeddingFeaturizer.TEST_WORD_EMBEDDING_FILE
        num_reps = 2
    else:
        dict_file = df.EmbeddingFeaturizer.WORD_EMBEDDING_FILE
        num_reps = 10

    relation_list_list = [extract_implicit_relations(dir, sense_lf) for dir in dir_list]

    word2vec = df.EmbeddingFeaturizer(dict_file)
    data_list = []
    for relation_list in relation_list_list:
        data = []
        data.extend(word2vec.additive_args(relation_list))
        data.extend(df.cached_features(relation_list, 'BaselineClassification'))
        data_list.append(data)
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, sense_lf)
    data_triplet = DataTriplet(data_list, [[x] for x in label_vectors], [label_alphabet])
    _net_experiment3_helper(experiment_name, json_file, num_reps, data_triplet, 
            use_linear=True, use_bilinear=True, use_hinge=True)
    _net_experiment3_helper(experiment_name, json_file, num_reps, data_triplet, 
            use_linear=True, use_bilinear=True, use_hinge=False)

def net_experiment3_cdssm(dir_list, args):
    """MOE = Baseline + CDSSM arg vectors
    """
    experiment_name = sys._getframe().f_code.co_name    
    json_file = set_logger(experiment_name)
    sense_lf = l.SecondLevelLabel()

    relation_list_list = [extract_implicit_relations(dir, sense_lf) for dir in dir_list]
    num_reps = 20

    data_list = []
    for relation_list in relation_list_list:
        data = []
        data.extend(df.cdssm_feature(relation_list))
        data.extend(df.cached_features(relation_list, 'BaselineClassification'))
        data_list.append(data)
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, sense_lf)
    data_triplet = DataTriplet(data_list, [[x] for x in label_vectors], [label_alphabet])
    _net_experiment3_helper(experiment_name, json_file, num_reps, data_triplet, 
            use_linear=True, use_bilinear=False, use_hinge=True)
    _net_experiment3_helper(experiment_name, json_file, num_reps, data_triplet, 
            use_linear=True, use_bilinear=False, use_hinge=False)


# net_experiment4 series
# Investigate the effectiveness of hidden layer in abstracting features
# We only look at ways of pooling {mean, max, sum, top}  and CDSSM
# proj_type must be one of {mean_pool, sum_pool, max_pool, top}
#
def net_experiment4_1(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    sense_lf = l.SecondLevelLabel()
    num_units = int(args[0])
    num_hidden_layers = int(args[1])
    projection = args[2]

    json_file = set_logger('%s_%sunits_%sh_%s' % \
            (experiment_name, num_units, num_hidden_layers, projection))

    relation_list_list = [extract_implicit_relations(dir, sense_lf) for dir in dir_list]
    word2vec_ff = util._get_word2vec_ff(num_units, projection)
    data_list = [word2vec_ff(relation_list) for relation_list in relation_list_list]
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, sense_lf)
    data_triplet = DataTriplet(data_list, [[x] for x in label_vectors], [label_alphabet])
    num_hidden_unit_list = [50, 200, 300, 400] 
    num_reps = 20
    for num_hidden_unit in num_hidden_unit_list:
        _net_experiment4_helper(json_file, num_hidden_layers, num_hidden_unit, num_reps,
                data_triplet, True)
        _net_experiment4_helper(json_file, num_hidden_layers, num_hidden_unit, num_reps,
                data_triplet, False)


def _net_experiment4_helper(json_file, num_hidden_layers, num_hidden_units, num_reps,
        data_triplet, use_hinge):
    n_epochs = 30
    learning_rate = 0.01
    lr_smoother = 0.01
    rng = np.random.RandomState(100)
    X_list = [T.matrix(), T.matrix()]
    if num_hidden_layers == 0:
        first_layer = LinearLayer(rng, 
            n_in_list=data_triplet.input_dimensions(),
            n_out=data_triplet.output_dimensions()[0],
            use_sparse=False, 
            X_list=X_list, 
            Y=T.lvector(),
            activation_fn=None if use_hinge else T.nnet.softmax)
    else:
        first_layer = LinearLayer(rng, 
            n_in_list=data_triplet.input_dimensions(),
            n_out=num_hidden_units,
            use_sparse=False, 
            X_list=X_list, 
            activation_fn=T.tanh)
    top_layer = first_layer
    layers = [first_layer]
    for i in range(num_hidden_layers):
        is_top_layer = i == (num_hidden_layers - 1)
        if is_top_layer:
            hidden_layer = LinearLayer(rng,
                n_in_list=[num_hidden_units], 
                n_out=data_triplet.output_dimensions()[0], 
                use_sparse=False, 
                X_list=[top_layer.activation], 
                Y=T.lvector(),
                activation_fn=None if use_hinge else T.nnet.softmax)
        else:
            hidden_layer = LinearLayer(rng,
                n_in_list=[num_hidden_units], 
                n_out=num_hidden_units, 
                use_sparse=False, 
                X_list=[top_layer.activation], 
                activation_fn=T.tanh)
        hidden_layer.params.extend(top_layer.params)
        layers.append(hidden_layer)
        top_layer = hidden_layer
    top_layer.input= X_list    
    trainer = AdagradTrainer(top_layer, 
            top_layer.hinge_loss if use_hinge else top_layer.crossentropy, 
            learning_rate, lr_smoother, data_triplet)

    for rep in xrange(num_reps):
        random_seed = rep
        rng = np.random.RandomState(random_seed)
        for layer in layers:
            layer.reset(rng)
        trainer.reset()
        
        minibatch_size = np.random.randint(20, 60)

        start_time = timeit.default_timer()
        best_iter, best_dev_acc, best_test_acc = \
                trainer.train_minibatch_triplet(minibatch_size, n_epochs, data_triplet)
        end_time = timeit.default_timer()
        print end_time - start_time 
        print best_iter, best_dev_acc, best_test_acc
        result_dict = {
                'test accuracy': best_test_acc,
                'best dev accuracy': best_dev_acc,
                'best iter': best_iter,
                'random seed': random_seed,
                'minibatch size': minibatch_size,
                'learning rate': learning_rate,
                'lr smoother': lr_smoother,
                'experiment name': experiment_name,
                'cost function': 'hinge' if use_hinge else 'crossentropy',
                'num hidden layers': num_hidden_layers,
                'num hidden units': num_hidden_units,
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))



# net_experiment5 series
# Trying to beat the baseline by MOE with neural networks 
# Especially the small feedforward nets with hidden layers
# and some SerialLSTM as well to see if the improvement is greater 
# and if we can have state-of-the-art results this way
#
def net_experiment5_1(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    sense_lf = l.SecondLevelLabel()
    num_units = int(args[0])
    num_hidden_layers = int(args[1])
    projection = args[2]



if __name__ == '__main__':
    experiment_name = sys.argv[1]
    mode = sys.argv[2]
    if mode == 'run':
        dir_list = ['conll15-st-05-19-15-train', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
    elif mode == 'dry':
        dir_list = ['conll15-st-05-19-15-dev', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
    globals()[experiment_name](dir_list, sys.argv[3:])
