"""Attention Mechanism experiments """
import json
import sys
import timeit

import numpy as np
import theano.tensor as T

import cognitive_disco.base_label_functions as l
from cognitive_disco.nets.learning import AdagradTrainer, DataTriplet
from cognitive_disco.data_reader import extract_implicit_relations
from cognitive_disco.nets.bilinear_layer import \
        BilinearLayer, LinearLayer, NeuralNet, make_multilayer_net
from cognitive_disco.nets.attention import \
        AttentionModelSimple, AttentionModelSimpleHiddenLayer
from cognitive_disco.nets.lstm import prep_serrated_matrix_relations
import cognitive_disco.nets.util as util


def att_experiment1(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    sense_lf = l.SecondLevelLabel()
    num_units = int(args[0])
    num_hidden_layers = int(args[1])
    use_att_hidden_layer = int(args[2])

    relation_list_list = [extract_implicit_relations(dir, sense_lf) 
            for dir in dir_list]
    wbm = util.get_wbm(num_units)
    data_list = []
    for relation_list in relation_list_list:
        data = prep_serrated_matrix_relations(relation_list, wbm, 50)
        data_list.append(data)
    label_vectors, label_alphabet = \
            util.label_vectorize(relation_list_list, sense_lf)
    data_triplet = DataTriplet(
            data_list, [[x] for x in label_vectors], [label_alphabet])
    num_reps = 10
    num_hidden_unit_list = [0] if num_hidden_layers == 0 \
            else [20, 50, 200, 300, 400] 

    json_file = util.set_logger('%s_%sunits_%sh_%s' % \
            (experiment_name, num_units, num_hidden_layers, use_att_hidden_layer))
    for num_hidden_units in num_hidden_unit_list:
        _att_experiment_ff_helper(experiment_name,
                json_file, data_triplet, wbm, num_reps, 
                use_att_hidden_layer,
                num_hidden_layers=num_hidden_layers, 
                num_hidden_units=num_hidden_units)


def _att_experiment_ff_helper(experiment_name, 
        json_file, data_triplet, wbm, num_reps, 
        use_att_hidden_layer, num_hidden_layers, num_hidden_units):
    rng = np.random.RandomState(100)

    if use_att_hidden_layer:
        arg1_model = AttentionModelSimpleHiddenLayer(rng, 
                wbm.num_units, num_hidden_units)
        arg2_model = AttentionModelSimpleHiddenLayer(rng, 
                wbm.num_units, num_hidden_units)
    else:
        arg1_model = AttentionModelSimple(rng, wbm.num_units)
        arg2_model = AttentionModelSimple(rng, wbm.num_units)
    _, predict_layers = make_multilayer_net(rng,
            n_in_list = [wbm.num_units, wbm.num_units],
            X_list = [arg1_model.activation, arg2_model.activation],
            Y=T.lvector(), use_sparse=False,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            num_output_units=data_triplet.output_dimensions()[0],
            output_activation_fn=T.nnet.softmax)
    layers = [arg1_model, arg2_model] + predict_layers
    nn = NeuralNet(layers)
    nn.input = arg1_model.input + arg2_model.input

    learning_rate = 0.001
    lr_smoother = 0.01
    trainer = AdagradTrainer(nn, nn.crossentropy, 
            learning_rate, lr_smoother, data_triplet, _make_givens)
    for rep in xrange(num_reps):
        random_seed = rep
        rng = np.random.RandomState(random_seed)
        nn.reset(rng)
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
                'cost function': 'crossentropy',
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))

def _make_givens(givens, input_vec, T_training_data, 
            output_vec, T_training_data_label, start_idx, end_idx):
    # first arg embedding and mask
    givens[input_vec[0]] = T_training_data[0][:,start_idx:end_idx, :]
    givens[input_vec[1]] = T_training_data[1][:,start_idx:end_idx]

    # second arg embedding and mask
    givens[input_vec[2]] = T_training_data[2][:,start_idx:end_idx, :]
    givens[input_vec[3]] = T_training_data[3][:,start_idx:end_idx]

    for i, output_var in enumerate(output_vec):
        givens[output_var] = T_training_data_label[i][start_idx:end_idx]

if __name__ == '__main__':
    experiment_name = sys.argv[1]
    mode = sys.argv[2]
    if mode == 'run':
        dir_list = ['conll15-st-05-19-15-train', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
    elif mode == 'dry':
        dir_list = ['conll15-st-05-19-15-dev', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
    globals()[experiment_name](dir_list, sys.argv[3:])
