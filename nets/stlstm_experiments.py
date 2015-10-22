import json
import sys
import timeit

import numpy as np
import theano.tensor as T

import cognitive_disco.base_label_functions as l
from cognitive_disco.data_reader import extract_implicit_relations

from cognitive_disco.nets.bilinear_layer import \
        LinearLayer, NeuralNet, LinearLayerTensorOutput
import cognitive_disco.nets.util as util
from cognitive_disco.nets.learning import AdagradTrainer, DataTriplet
from cognitive_disco.nets.lstm import \
        BinaryTreeLSTM, \
        prep_stlstm_serrated_matrix_relations

def net_experiment_stlstm(dir_list, args):
    num_units = int(args[0])
    num_hidden_layers = int(args[1])
    proj_type = args[2]
    experiment_name = sys._getframe().f_code.co_name    
    json_file = util.set_logger('%s__%sunits_%sh_%s' % \
            (experiment_name,  num_units, num_hidden_layers, proj_type))
    sense_lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, sense_lf) 
            for dir in dir_list]
    wbm = util.get_wbm(num_units)
    labels = ['NP', 'VP' , 'S', 'PP', 'SBAR', 'ADVP', 'ADJP', 
            'NP_', 'VP_', 'S_', 'PP_', 'SBAR_', 'ADVP_', 'ADJP_', 
            'OTHERS']
    node_label_alphabet = {}
    for label in labels:
        node_label_alphabet[label] = len(node_label_alphabet)

    data_list = []
    node_label_tuple_triplet = []
    for relation_list in relation_list_list:
        data, node_labels_tuple = \
                prep_stlstm_serrated_matrix_relations(relation_list, wbm, 30, node_label_alphabet)
        data_list.append(data)
        node_label_tuple_triplet.append(node_labels_tuple)
    label_vector_triplet, label_alphabet = \
            util.label_vectorize(relation_list_list, sense_lf)

    outputs = []
    for x,y in zip(node_label_tuple_triplet, label_vector_triplet):
        d = [m for m in x]
        d.append(y)
        outputs.append(d)
    output_alphabet = [node_label_alphabet, node_label_alphabet, label_alphabet]
    data_triplet = DataTriplet(data_list, outputs, output_alphabet)

    num_reps = 15
    num_hidden_unit_list = [0] if num_hidden_layers == 0 \
            else [50, 200, 300, 400] 

    for num_hidden_units in num_hidden_unit_list:
        _net_experiment_stlstm_helper(experiment_name, 
                json_file, data_triplet, wbm, num_reps, 
                num_hidden_layers=num_hidden_layers, 
                num_hidden_units=num_hidden_units, 
                proj_type=proj_type,
                )

def _construct_net(data_triplet, wbm, 
        num_hidden_layers, num_hidden_units, proj_type):
    rng = np.random.RandomState(100)
    arg1_model = BinaryTreeLSTM(rng, wbm.num_units)
    arg2_model = BinaryTreeLSTM(rng, wbm.num_units)

    arg1_node_label_layer = LinearLayerTensorOutput(rng, 
            n_in=wbm.num_units, 
            n_out=data_triplet.output_dimensions()[0],
            X=arg1_model.h)
    arg2_node_label_layer = LinearLayerTensorOutput(rng, 
            n_in=wbm.num_units, 
            n_out=data_triplet.output_dimensions()[1],
            X=arg2_model.h)

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
    label_output_layer = LinearLayer(rng, 
            n_in_list=n_in_list,
            n_out=data_triplet.output_dimensions()[2],
            use_sparse=False,
            X_list=X_list,
            Y=T.lvector(),
            activation_fn=T.nnet.softmax)

    nn = NeuralNet()
    layers = [arg1_model, arg2_model, 
            arg1_node_label_layer, arg2_node_label_layer, label_output_layer] \
                    + hidden_layers
    for layer in layers:
        nn.params.extend(layer.params)
    nn.layers = layers

    nn.input.extend(arg1_model.input)
    nn.input.extend(arg2_model.input)
    nn.output.extend(arg1_node_label_layer.output + 
            arg2_node_label_layer.output + 
            label_output_layer.output)
    nn.predict = label_output_layer.predict
    nn.crossentropy = label_output_layer.crossentropy + \
            0.5 * arg1_node_label_layer.crossentropy + \
            0.5 * arg2_node_label_layer.crossentropy
    
    nn.misc_function = arg1_node_label_layer.miscs + arg2_node_label_layer.miscs
    """
    nn.crossentropy = arg2_node_label_layer.crossentropy + arg1_node_label_layer.crossentropy
    nn.params = arg2_node_label_layer.params + arg1_node_label_layer.params
    nn.misc_function = [arg1_node_label_layer.W[0:3,0:3], arg2_node_label_layer.W[0:3,0:3]]
    nn.crossentropy = label_output_layer.crossentropy
    if len(hidden_layers) > 0:
        nn.layers = [arg1_model, arg2_model, hidden_layers[0], label_output_layer]
        nn.params = arg1_model.params + arg2_model.params + hidden_layers[0].params + label_output_layer.params
    else:
        nn.layers = [arg1_model, arg2_model, label_output_layer]
        nn.params = arg1_model.params + arg2_model.params + label_output_layer.params
    nn.misc_function = [label_output_layer.predict[0], 
            label_output_layer.W_list[0][0:3,0:3],
            arg1_node_label_layer.W[0:3,0:3],
            label_output_layer.activation]
    """

    return nn

def _net_experiment_stlstm_helper(experiment_name, json_file, data_triplet, wbm, 
        num_reps, num_hidden_layers, num_hidden_units, proj_type):
    nn = _construct_net(data_triplet, wbm, 
            num_hidden_layers, num_hidden_units, proj_type)
    learning_rate = 0.01
    lr_smoother = 0.01
    trainer = AdagradTrainer(nn, nn.crossentropy,
            learning_rate, lr_smoother, data_triplet, 
            BinaryTreeLSTM.make_givens, nn.misc_function)
    
    for rep in xrange(num_reps):
        random_seed = rep
        rng = np.random.RandomState(random_seed)
        for layer in nn.layers:
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
                'cost function': 'crossentropy',
                'projection' : proj_type,
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))
