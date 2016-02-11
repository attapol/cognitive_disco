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
        InputLayer, make_multilayer_net_from_layers
from cognitive_disco.nets.attention import \
        AttentionModelSimple, AttentionLSTM
from cognitive_disco.nets.lstm import prep_serrated_matrix_relations
import cognitive_disco.nets.util as util


def att_experiment1(dir_list, args):
    experiment_name = sys._getframe().f_code.co_name    
    sense_lf = l.SecondLevelLabel()
    num_units = int(args[0])
    num_hidden_layers = int(args[1])
    num_att_hidden_layer = int(args[2])
    dropout_arg = args[3]
    assert(dropout_arg =='d' or dropout_arg =='n')
    dropout = True if dropout_arg == 'd' else False
    

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
            else [300, 400, 600, 800] 

    json_file = util.set_logger('%s_%sunits_%sh_%sh_%s' % \
            (experiment_name, num_units, 
                num_hidden_layers, num_att_hidden_layer, dropout_arg))
    for num_hidden_units in num_hidden_unit_list:
        _att_experiment_ff_helper(experiment_name, AttentionModelSimple,
                json_file, data_triplet, wbm, num_reps, 
                num_att_hidden_layer, num_hidden_layers, num_hidden_units,
                dropout)

def att_experiment2(dir_list, args):
    """Simple feedforward net without the attention model. 

    This is just to test out the improvement we can gain from applying dropout
    """
    experiment_name = sys._getframe().f_code.co_name    
    sense_lf = l.SecondLevelLabel()
    num_units = int(args[0])
    num_hidden_layers = int(args[1])
    dropout_arg = args[2]
    assert(dropout_arg =='d' or dropout_arg =='n')
    dropout = True if dropout_arg == 'd' else False

    relation_list_list = [extract_implicit_relations(dir, sense_lf) 
            for dir in dir_list]
    wbm = util.get_wbm(num_units)
    data_list = []
    word2vec_ff = util._get_word2vec_ff(num_units, 'sum_pool')
    data_list = [word2vec_ff(relation_list) \
            for relation_list in relation_list_list]
    label_vectors, label_alphabet = \
            util.label_vectorize(relation_list_list, sense_lf)
    data_triplet = DataTriplet(
            data_list, [[x] for x in label_vectors], [label_alphabet])

    num_reps = 10
    num_hidden_unit_list = [0] if num_hidden_layers == 0 \
            else [1000, 200, 300, 400, 600] 

    json_file = util.set_logger('%s_%sunits_%sh_%s' % \
            (experiment_name, num_units, num_hidden_layers, dropout_arg))
    for num_hidden_units in num_hidden_unit_list:
        _train_feedforward_net(experiment_name,
                json_file, data_triplet, wbm, num_reps, 
                num_hidden_layers, num_hidden_units, dropout)

def att_experiment3(dir_list, args):
    """LSTM Attention Mechanism for feedforward nets
    """
    experiment_name = sys._getframe().f_code.co_name    
    num_units = int(args[0])
    num_hidden_layers = int(args[1])
    num_att_hidden_layer = int(args[2])
    dropout_arg = args[3]
    assert(dropout_arg =='d' or dropout_arg =='n')
    dropout = True if dropout_arg == 'd' else False
    wbm = util.get_wbm(num_units)
    data_triplet = util.get_data_srm(dir_list, wbm)
    num_reps = 10
    num_hidden_unit_list = [0] if num_hidden_layers == 0 \
            else [300, 400, 600, 800] 

    json_file = util.set_logger('%s_%sunits_%sh_%sh_%s' % \
            (experiment_name, num_units, 
                num_hidden_layers, num_att_hidden_layer, dropout_arg))
    for num_hidden_units in num_hidden_unit_list:
        _att_experiment_ff_helper(experiment_name, AttentionLSTM,
                json_file, data_triplet, wbm, num_reps, 
                num_att_hidden_layer, num_hidden_layers, num_hidden_units,
                dropout)

def _train_feedforward_net(experiment_name,
        json_file, data_triplet, wbm, num_reps, 
        num_hidden_layers, num_hidden_units, dropout):
    rng = np.random.RandomState(100)
    arg1_model = InputLayer(rng, wbm.num_units, False)
    arg2_model = InputLayer(rng, wbm.num_units, False)

    nn, all_layers = make_multilayer_net_from_layers(
            input_layers=[arg1_model, arg2_model],
            Y=T.lvector(), use_sparse=False,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            num_output_units=data_triplet.output_dimensions()[0],
            output_activation_fn=T.nnet.softmax,
            dropout=dropout)

    learning_rate = 0.01
    lr_smoother = 0.01
    trainer = AdagradTrainer(nn, nn.crossentropy, 
            learning_rate, lr_smoother, data_triplet, util.make_givens)
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
                'dropout': dropout
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))


def _att_experiment_ff_helper(experiment_name, attention_model,
        json_file, data_triplet, wbm, num_reps, 
        num_att_hidden_layer, num_hidden_layers, num_hidden_units,
        dropout):
    rng = np.random.RandomState(100)

    arg1_model = attention_model(rng, 
            wbm.num_units, num_att_hidden_layer, num_hidden_units,
            dropout=False)
    arg2_model = attention_model(rng, 
            wbm.num_units, num_att_hidden_layer, num_hidden_units,
            dropout=False)
    nn, all_layers = make_multilayer_net_from_layers(
            input_layers=[arg1_model, arg2_model],
            Y=T.lvector(), use_sparse=False,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            num_output_units=data_triplet.output_dimensions()[0],
            output_activation_fn=T.nnet.softmax,
            dropout=dropout)
    nn.input = arg1_model.input + arg2_model.input
    #print 'before num params %s' % len(nn.params)
    #nn.params = nn.params[(len(arg1_model.params) + len(arg2_model.params)):]
    #print 'after num params %s' % len(nn.params)

    learning_rate = 0.001
    lr_smoother = 0.01
    trainer = AdagradTrainer(nn, nn.crossentropy, 
            learning_rate, lr_smoother, data_triplet, 
            util.make_givens_srm)
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
        print 'Training process takes %s seconds' % end_time - start_time 
        print 'Best iteration is %s;' % best_iter + \
                'Best dev accuracy = %s' % best_dev_acc + \
                'Test accuracy =%s' % best_test_acc
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
                'dropout': dropout
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))


def get_net():
    """This is for debugging purposes. Should be torn apart to your taste

    This way you can investigate the activation as you pass the data through.
    """
    num_units = 50
    sense_lf = l.SecondLevelLabel()
    num_hidden_layers = 1
    dropout = True
    num_hidden_units = 400
    num_att_hidden_layer = 1

    dir_list = ['conll15-st-05-19-15-train', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
    dir_list = ['conll15-st-05-19-15-dev', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
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

    rng = np.random.RandomState(100)

    arg1_model = AttentionModelSimple(rng, 
            wbm.num_units, num_att_hidden_layer, num_hidden_units,
            dropout=False)
    arg2_model = AttentionModelSimple(rng, 
            wbm.num_units, num_att_hidden_layer, num_hidden_units,
            dropout=False)
    nn, all_layers = make_multilayer_net_from_layers(
            input_layers=[arg1_model, arg2_model],
            Y=T.lvector(), use_sparse=False,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            num_output_units=data_triplet.output_dimensions()[0],
            output_activation_fn=T.nnet.softmax,
            dropout=dropout)
    nn.input = arg1_model.input + arg2_model.input
    print 'before num params %s' % len(nn.params)
    arg1_model.params[-1].set_value([100])
    arg2_model.params[-1].set_value([100])
    nn.params = nn.params[(len(arg1_model.params) + len(arg2_model.params)):]
    print 'after num params %s' % len(nn.params)
    return nn, data_triplet

if __name__ == '__main__':
    experiment_name = sys.argv[1]
    mode = sys.argv[2]
    if mode == 'run':
        dir_list = ['conll15-st-05-19-15-train', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
    elif mode == 'dry':
        dir_list = ['conll15-st-05-19-15-dev', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
    globals()[experiment_name](dir_list, sys.argv[3:])
