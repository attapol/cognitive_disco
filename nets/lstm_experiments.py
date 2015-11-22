import json
import sys
import timeit

import numpy as np
import theano.tensor as T

import cognitive_disco.base_label_functions as l
from cognitive_disco.nets.learning import AdagradTrainer, DataTriplet
from cognitive_disco.data_reader import extract_implicit_relations
from cognitive_disco.nets.bilinear_layer import \
        BilinearLayer, LinearLayer, NeuralNet
from cognitive_disco.nets.lstm import \
        SerialLSTM, prep_serrated_matrix_relations, \
        BinaryTreeLSTM, prep_tree_lstm_serrated_matrix_relations
import cognitive_disco.nets.util as util

def net_experiment_lstm(dir_list, args):
    """

    Args : Five required arguments
        linear (l) or bilinear(bl)
        num units is the number of the units in the embedding (NOT HIDDEN LAYERS)
        num hidden layers is the number of hidden layers
        proj_type must be one of {mean_pool, sum_pool, max_pool, top}
        shared 
    """
    assert(len(args) == 5)

    if args[0] == 'bl':
        use_bl = True
    elif args[0] == 'l':
        use_bl = False
    else:
        raise ValueError('First argument must be l or bl')
    num_units = int(args[1])
    num_hidden_layers = int(args[2])
    proj_type = args[3]
    if args[4] == 'shared':
        arg_shared_weights = True
    elif args[4] == 'noshared':
        arg_shared_weights = False
    else:
        raise ValueError('Last argument must be shared or noshared')

    experiment_name = sys._getframe().f_code.co_name    
    json_file = util.set_logger('%s_%s_%sunits_%sh_%s_%s' % \
            (experiment_name, args[0], num_units, 
                num_hidden_layers, proj_type, args[4]))
    sense_lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, sense_lf) 
            for dir in dir_list]

    wbm = util.get_wbm(num_units)
    data_list = []
    for relation_list in relation_list_list:
        data = prep_serrated_matrix_relations(relation_list, wbm, 30)
        data_list.append(data)
    label_vectors, label_alphabet = \
            util.label_vectorize(relation_list_list, sense_lf)
    data_triplet = DataTriplet(
            data_list, [[x] for x in label_vectors], [label_alphabet])

    num_reps = 10
    num_hidden_unit_list = [0] if num_hidden_layers == 0 \
            else [20, 50, 200, 300, 400] 
    for num_hidden_units in num_hidden_unit_list:
        _net_experiment_lstm_helper(experiment_name,
                json_file, data_triplet, wbm, num_reps, 
                SerialLSTM,
                num_hidden_layers=num_hidden_layers, 
                num_hidden_units=num_hidden_units, 
                use_hinge=False, 
                proj_type=proj_type,
                use_bl=use_bl,
                arg_shared_weights=arg_shared_weights
                )

def net_experiment_tree_lstm(dir_list, args):
    """
    Args : Five required arguments
        linear (l) or bilinear(bl)
        num units is the number of the units in the embedding (NOT HIDDEN LAYERS)
        num hidden layers is the number of hidden layers
        proj_type must be one of {mean_pool, sum_pool, max_pool, top}
        shared 
    """
    assert(len(args) >= 5)

    if args[0] == 'bl':
        use_bl = True
    elif args[0] == 'l':
        use_bl = False
    else:
        raise ValueError('First argument must be l or bl')
    num_units = int(args[1])
    num_hidden_layers = int(args[2])
    proj_type = args[3]
    if args[4] == 'shared':
        arg_shared_weights = True
    elif args[4] == 'noshared':
        arg_shared_weights = False
    else:
        raise ValueError('Last argument must be shared or noshared')

    if len(args) == 6 and args[5] == 'left':
        all_left_branching = True
    else:
        all_left_branching = False

    experiment_name = sys._getframe().f_code.co_name    
    if all_left_branching:
        json_file = util.set_logger('%s_%s_%sunits_%sh_%s_%s_left' % \
                (experiment_name, args[0], num_units, 
                    num_hidden_layers, proj_type, args[4]))
    else:
        json_file = util.set_logger('%s_%s_%sunits_%sh_%s_%s' % \
                (experiment_name, args[0], num_units, 
                    num_hidden_layers, proj_type, args[4]))
    sense_lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, sense_lf) 
            for dir in dir_list]

    wbm = util.get_wbm(num_units)
    data_list = []
    for relation_list in relation_list_list:
        data = prep_tree_lstm_serrated_matrix_relations(
                relation_list, wbm, 35)
        data_list.append(data)

    label_vectors, label_alphabet = \
            util.label_vectorize(relation_list_list, sense_lf)
    data_triplet = DataTriplet(
            data_list, [[x] for x in label_vectors], [label_alphabet])

    num_reps = 15
    num_hidden_unit_list = [0] if num_hidden_layers == 0 \
            else [50, 200, 300, 400] 
    for num_hidden_units in num_hidden_unit_list:
        _net_experiment_lstm_helper(experiment_name,
                json_file, data_triplet, wbm, num_reps, 
                BinaryTreeLSTM,
                num_hidden_layers=num_hidden_layers, 
                num_hidden_units=num_hidden_units, 
                use_hinge=False, 
                proj_type=proj_type,
                use_bl=use_bl,
                arg_shared_weights=arg_shared_weights
                )

def _net_experiment_lstm_helper(experiment_name,
        json_file, data_triplet, wbm, num_reps, 
        LSTMModel, num_hidden_layers, num_hidden_units, use_hinge, proj_type, 
        use_bl, arg_shared_weights):

    rng = np.random.RandomState(100)
    arg1_model = LSTMModel(rng, wbm.num_units)
    if arg_shared_weights:
        arg2_model = LSTMModel(rng, wbm.num_units, 
                W=arg1_model.W, U=arg1_model.U, b=arg1_model.b)
    else:
        arg2_model = LSTMModel(rng, wbm.num_units)

    if proj_type == 'max_pool':
        proj_variables = [arg1_model.max_pooled_h, arg2_model.max_pooled_h]
    elif proj_type == 'mean_pool':
        proj_variables = [arg1_model.mean_pooled_h, arg2_model.mean_pooled_h]
    elif proj_type == 'sum_pool':
        proj_variables = [arg1_model.sum_pooled_h, arg2_model.sum_pooled_h]
    elif proj_type == 'top':
        proj_variables = [arg1_model.top_h, arg2_model.top_h]
    else:
        raise ValueError('Invalid projection type: %s' % proj_type)

    hidden_layers = []    
    if use_bl:
        output_layer = BilinearLayer(rng, 
                n_in1=wbm.num_units,
                n_in2=wbm.num_units,
                n_out=data_triplet.output_dimensions()[0],
                X1=proj_variables[0],
                X2=proj_variables[1],
                Y=T.lvector(),
                activation_fn=None if use_hinge else T.nnet.softmax)
    else:
        n_in_list = [wbm.num_units, wbm.num_units]
        X_list = proj_variables
        for i in range(num_hidden_layers):
            hidden_layer = LinearLayer(rng,
                n_in_list=n_in_list, 
                n_out=num_hidden_units,
                use_sparse=False, 
                X_list=X_list, 
                activation_fn=T.tanh)
            n_in_list = [num_hidden_units]
            X_list = [hidden_layer.activation]
            hidden_layers.append(hidden_layer)
        output_layer = LinearLayer(rng, 
                n_in_list=n_in_list,
                n_out=data_triplet.output_dimensions()[0],
                use_sparse=False,
                X_list=X_list,
                Y=T.lvector(),
                activation_fn=None if use_hinge else T.nnet.softmax)

    nn = NeuralNet()
    layers = [arg1_model, arg2_model, output_layer] + hidden_layers
    nn.params.extend(arg1_model.params)
    if not arg_shared_weights:
        nn.params.extend(arg2_model.params)
    nn.params.extend(output_layer.params)
    for hidden_layer in hidden_layers:
        nn.params.extend(hidden_layer.params)
    nn.layers = layers

    nn.input.extend(arg1_model.input)
    nn.input.extend(arg2_model.input)
    nn.output.extend(output_layer.output)
    nn.predict = output_layer.predict
    nn.hinge_loss = output_layer.hinge_loss
    nn.crossentropy = output_layer.crossentropy

    learning_rate = 0.001
    lr_smoother = 0.01

    trainer = AdagradTrainer(nn,
            nn.hinge_loss if use_hinge else nn.crossentropy,
            learning_rate, lr_smoother, data_triplet, LSTMModel.make_givens)
    
    for rep in xrange(num_reps):
        random_seed = rep
        rng = np.random.RandomState(random_seed)
        for layer in layers:
            layer.reset(rng)
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
                'num hidden layers': num_hidden_layers,
                'cost function': 'hinge loss' if use_hinge else 'crossentropy',
                'projection' : proj_type,
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))


