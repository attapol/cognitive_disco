"""Mixture Experiments


Using the surface features in combination with the continuous features
induced by word embeddings.

"""
import json
import sys
import timeit

import cognitive_disco.base_label_functions as l
import cognitive_disco.nets.util as util
from cognitive_disco.data_reader import extract_implicit_relations
from cognitive_disco.nets.learning import AdagradTrainer, DataTriplet
from cognitive_disco.nets.bilinear_layer import \
        LinearLayer, MixtureOfExperts, NeuralNet

from theano import config
import theano.sparse
import theano.tensor as T
import numpy as np
from scipy.sparse import coo_matrix

def net_mixture_experiment1(dir_list, args):
    """Experiment 1

    Read the sparse feature matrices from the files
    Read the parameters from the file (pretrained model) or not

    Use Mixture of Experts model
    The gating units do not have any hidden layers.
    """
    sparse_feature_file = args[0]
    num_units = int(args[1])
    cont_num_hidden_layers = int(args[2])
    sparse_num_hidden_layers = int(args[3])
    proj_type = args[4]

    experiment_name = sys._getframe().f_code.co_name    
    json_file = util.set_logger('%s_%sunits_%sh_%sh_%s' % \
            (experiment_name,  num_units, 
                cont_num_hidden_layers,
                sparse_num_hidden_layers, proj_type))
    sense_lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, sense_lf) 
            for dir in dir_list]

    num_features = 250000
    id_to_sfv = read_sparse_vectors(dir_list, sparse_feature_file)
    sfv_data_list = [get_sfv(relation_list, id_to_sfv, num_features) 
            for relation_list in relation_list_list]

    word2vec_ff = util._get_word2vec_ff(num_units, proj_type)
    word2vec_data_list = [word2vec_ff(relation_list) 
            for relation_list in relation_list_list]

    data_list = [[x] + y for x, y in zip(sfv_data_list, word2vec_data_list)]

    label_vector_triplet, label_alphabet = \
            util.label_vectorize(relation_list_list, sense_lf)
    data_triplet = DataTriplet(
            data_list, [[x] for x in label_vector_triplet], [label_alphabet])

    num_hidden_unit_list = [50, 200, 300, 400] 
    num_reps = 20
    for num_hidden_unit in num_hidden_unit_list:
        _net_mixture_experiment_helper(experiment_name, 
                json_file, data_triplet, num_reps, 
                sparse_num_hidden_layers, cont_num_hidden_layers,
                num_hidden_unit, proj_type)



def _add_hidden_layers(layer, num_hidden_units, num_hidden_layers, num_out):
    top_layer = layer  
    rng = layer.rng
    layers = [layer]
    for i in range(num_hidden_layers):
        is_last_layer = i == (num_hidden_layers - 1)
        if is_last_layer:
            hidden_layer = LinearLayer(rng,
                    n_in_list=[num_hidden_units],
                    n_out=num_out,
                    use_sparse=False,
                    X_list=[top_layer.activation],
                    Y=T.lvector(),
                    activation_fn=T.nnet.softmax)
        else:
            hidden_layer = LinearLayer(rng, 
                    n_in_list=[num_hidden_units],
                    n_out=num_hidden_units,
                    use_sparse=False,
                    X_list=[top_layer.activation],
                    activation_fn=T.tanh)
        layers.append(hidden_layer)
        top_layer = hidden_layer
    return layers

def _make_module(rng, n_in_list, X_list, use_sparse, 
        num_hidden_layers, num_hidden_units, num_output_units):
    if num_hidden_layers == 0:
        n_out = num_output_units
        activation_fn = T.nnet.softmax
        Y = T.lvector()
    else:
        n_out = num_hidden_units
        activation_fn = T.tanh
        Y = None
    layer = LinearLayer(rng, n_in_list=n_in_list, n_out=n_out, 
            use_sparse=True, X_list=X_list, Y=Y, activation_fn=activation_fn)
    layers = _add_hidden_layers(layer, 
            num_hidden_units, num_hidden_layers, num_output_units)
    return NeuralNet(layers)

def _net_mixture_experiment_helper(experiment_name, 
        json_file, data_triplet, num_reps, sparse_num_hidden_layers, 
        cont_num_hidden_layers, num_hidden_units, proj_type):
    rng = np.random.RandomState(100)
    learning_rate = 0.001
    lr_smoother = 0.01

    # the first one should be sparse
    X_list = [theano.sparse.csr_matrix(), T.matrix(), T.matrix()]
    sf_net = _make_module(rng, 
            n_in_list=data_triplet.input_dimensions()[0:1],
            X_list=X_list[0:1], use_sparse=True,
            num_hidden_layers=sparse_num_hidden_layers, 
            num_hidden_units=num_hidden_units, 
            num_output_units=data_triplet.output_dimensions()[0])
    word2vec_net = _make_module(rng, 
            n_in_list=data_triplet.input_dimensions()[1:],
            X_list=X_list[1:], use_sparse=False,
            num_hidden_layers=cont_num_hidden_layers, 
            num_hidden_units=num_hidden_units, 
            num_output_units=data_triplet.output_dimensions()[0])

    moe = MixtureOfExperts(rng,
            n_in_list=data_triplet.input_dimensions(),
            expert_list=[sf_net, word2vec_net],
            X_list=X_list, 
            Y=T.lvector())

    trainer = AdagradTrainer(moe, moe.crossentropy, 
            learning_rate, lr_smoother, data_triplet, _make_givens)
    for rep in xrange(num_reps):
        random_seed = rep
        rng = np.random.RandomState(random_seed)
        moe.reset(rng)
        trainer.reset()

        minibatch_size = np.random.randint(20, 60)
        n_epochs = 50

        start_time = timeit.default_timer()
        best_iter, best_dev_acc, best_test_acc = \
                trainer.train_minibatch_triplet(minibatch_size, n_epochs)
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
                'num hidden units': num_hidden_units,
                'sparse num hidden layers': sparse_num_hidden_layers,
                'continuous num hidden layers': cont_num_hidden_layers,
                'cost function': 'crossentropy',
                'projection' : proj_type,
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))

def _make_givens(givens, input_vec, T_training_data, 
            output_vec, T_training_data_label, start_idx, end_idx):
    for i, input_var in enumerate(input_vec):
        givens[input_var] = T_training_data[i][start_idx:end_idx]

    for i, output_var in enumerate(output_vec):
        givens[output_var] = T_training_data_label[i][start_idx:end_idx]

def read_sparse_vectors(dir_list, sparse_feature_file):
    id_to_sfv = {}
    for dir in dir_list:
        file_name = '%s/%s' % (dir, sparse_feature_file)
        for line in open(file_name):
            split_line = line.strip().split()
            id_to_sfv[split_line[0]] = [int(x) for x in split_line[1:]]
    return id_to_sfv

def get_sfv(relation_list, id_to_sfv, num_features=250000):
    rows = []
    columns = []
    data = []
    for i, relation in enumerate(relation_list):
        relation_id = relation.doc_relation_id
        columns += id_to_sfv[relation_id]
        for j in xrange(len(id_to_sfv[relation_id])):
            rows.append(i)
            data.append(1)
    return coo_matrix(
            (data, (rows, columns)),
            shape=(len(relation_list), num_features), 
            dtype=config.floatX).tocsr()
