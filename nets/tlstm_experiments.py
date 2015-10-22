import timeit

from cognitive_disco.nets.tlstm import \
        BinaryForestLSTM, prep_trees
import numpy as np
import theano.sparse
import theano.tensor as T

from cognitive_disco.nets.learning import AdagradTrainer, DataTriplet
from cognitive_disco.data_reader import extract_implicit_relations
from cognitive_disco.nets.bilinear_layer import \
        LinearLayer, NeuralNet
import cognitive_disco.nets.util as util

def net_experiment_tlstm(dir_list, args):
    """Tree-structured LSTM experiment version 2

    This version is different from net_experiment_tree_lstm in that 
    you use BinaryForestLSTM instead. The tree structures along with 
    the data themselvesa are encoded internally in the model. 
    This way, we can be sure that the algorithm is doing the right thing.
    Scan loop is very slow as well so I hope that this works better. 
    ipython experiments.py net_experiment_tlstm l 50 1 mean_pool
    """
    assert(len(args) >= 4)

    if args[0] == 'bl':
        use_bl = True
        raise ValueError('bilinear model is not supported yet')
    elif args[0] == 'l':
        use_bl = False
    else:
        raise ValueError('First argument must be l or bl')
    num_units = int(args[1])
    num_hidden_layers = int(args[2])
    proj_type = args[3]
    if len(args) == 5 and args[4] == 'left':
        all_left_branching = True
        print 'use left branching trees'
    else:
        all_left_branching = False

    experiment_name = sys._getframe().f_code.co_name    
    if all_left_branching:
        name_file = '%s_%s_%sunits_%sh_%s_left' % \
                (experiment_name, args[0], num_units, 
                    num_hidden_layers, proj_type)
        json_file = util.set_logger(name_file)
        model_file = name_file + '.model'
    else:
        name_file = '%s_%s_%sunits_%sh_%s' % \
                (experiment_name, args[0], num_units, 
                    num_hidden_layers, proj_type)
        json_file = util.set_logger(name_file)
        model_file = name_file + '.model'
    sense_lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, sense_lf)[0:5]
            for dir in dir_list]
    wbm = util.get_wbm(num_units)

    data_list = []
    for relation_list in relation_list_list:
        data = prep_trees(relation_list)
        data_list.append(data)

    label_vectors, label_alphabet = \
            util.label_vectorize(relation_list_list, sense_lf)
    data_triplet = DataTriplet(
            data_list, [[x] for x in label_vectors], [label_alphabet])

    num_reps = 15
    num_hidden_unit_list = [0] if num_hidden_layers == 0 \
            else [50, 200, 300, 400] 
    for num_hidden_units in num_hidden_unit_list:
        _net_experiment_tlstm_helper(json_file, model_file,
                data_triplet, wbm, num_reps, 
                num_hidden_layers=num_hidden_layers, 
                num_hidden_units=num_hidden_units, 
                use_hinge=False, 
                proj_type=proj_type,
                )

def _net_experiment_tlstm_helper(json_file, model_file, 
        data_triplet, wbm, num_reps, 
        num_hidden_layers, num_hidden_units, use_hinge, proj_type):
    nn = _make_tlstm_net(data_triplet.training_data, wbm, 
            data_triplet.output_dimensions()[0], num_hidden_layers, 
            num_hidden_units, use_hinge, proj_type)

    start_time = timeit.default_timer()
    theano.function(inputs=nn.input+nn.output, outputs=nn.crossentropy)
    end_time = timeit.default_timer()
    num_data = len(data_triplet.training_data_label[0])
    print 'crossentropy function for %s instances take %s seconds' % (num_data, end_time - start_time )

    learning_rate = 0.001
    lr_smoother = 0.01
    indexed_data_triplet = DataTriplet(
            data_list=[ 
                [np.arange(len(data_triplet.training_data_label[0]))], 
                [np.arange(len(data_triplet.dev_data_label[0]))],
                [np.arange(len(data_triplet.test_data_label[0]))]
                ],
            label_vectors=[data_triplet.training_data_label,
                data_triplet.dev_data_label,
                data_triplet.test_data_label],
            label_alphabet_list=data_triplet.label_alphabet_list)

    #print nn.input
    #f = theano.function(inputs=nn.input[0:1] + nn.output, outputs=nn.crossentropy)
    #print f(np.array([2]), np.array([2]))

    start_time = timeit.default_timer()
    trainer = AdagradTrainer(nn,
            nn.hinge_loss if use_hinge else nn.crossentropy,
            learning_rate, lr_smoother, indexed_data_triplet, 
            BinaryForestLSTM.make_givens)
    end_time = timeit.default_timer()
    num_data = len(indexed_data_triplet.training_data_label[0])
    print '%s instances take %s seconds' % (num_data, end_time - start_time )
    return
    
    dev_model = _copy_tlstm_net(data_triplet.dev_data, nn, proj_type)
    test_model = _copy_tlstm_net(data_triplet.test_data, nn, proj_type)

    dev_accuracy = T.mean(T.eq(dev_model.output[-1], dev_model.predict[-1]))
    trainer.dev_eval_function = \
            theano.function(inputs=dev_model.input + dev_model.output, 
                    outputs=[dev_accuracy, dev_model.crossentropy],
                    on_unused_input='warn')

    test_accuracy = T.mean(T.eq(test_model.output[-1], test_model.predict[-1]))
    trainer.test_eval_function = \
            theano.function(inputs=test_model.input + test_model.output, 
                    outputs=[test_accuracy, test_model.crossentropy],
                    on_unused_input='warn')
    #with open(model_file, 'w') as f:
        #sys.setrecursionlimit(5000000)
        #cPickle.dump(trainer, f)

    for rep in xrange(num_reps):
        random_seed = rep
        rng = np.random.RandomState(random_seed)
        for layer in nn.layers:
            layer.reset(rng)
        trainer.reset()
        
        minibatch_size = np.random.randint(20, 60)
        minibatch_size = 1
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
                'experiment name': model_file,
                'num hidden units': num_hidden_units,
                'num hidden layers': num_hidden_layers,
                'cost function': 'hinge loss' if use_hinge else 'crossentropy',
                'projection' : proj_type,
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))

def _copy_tlstm_net(data_list, nn, proj_type):
    arg1_model = nn.layers[0].copy(data_list[0])
    arg2_model = nn.layers[1].copy(data_list[1])
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
    X_list = proj_variables
    new_hidden_layers = []
    for hidden_layer in nn.layers[2:-1]:
       new_hidden_layer = hidden_layer.copy(X_list) 
       X_list = [new_hidden_layer.activation]
       new_hidden_layers.append(new_hidden_layer)
    output_layer = nn.layers[-1].copy(X_list)

    new_nn = NeuralNet()
    layers = [arg1_model, arg2_model] + new_hidden_layers +  [output_layer] 
    new_nn.layers = layers

    new_nn.input.extend(arg1_model.input)
    new_nn.output.extend(output_layer.output)
    new_nn.predict = output_layer.predict
    new_nn.hinge_loss = output_layer.hinge_loss
    new_nn.crossentropy = output_layer.crossentropy
    return new_nn

def _make_tlstm_net(data_list, wbm, num_output_units,
        num_hidden_layers, num_hidden_units, use_hinge, proj_type):

    rng = np.random.RandomState(100)
    indices = T.lvector()

    arg1_model = BinaryForestLSTM(data_list[0], rng, wbm, X_list=[indices])
    arg2_model = BinaryForestLSTM(data_list[1], rng, wbm, X_list=[indices])
    #f = theano.function(inputs=arg1_model.input, 
           #outputs=[arg1_model.max_pooled_h, arg1_model.all_max_pooled_h.shape])
    #print f(np.array([2]))
    #f = theano.function(inputs=arg1_model.input, 
           #outputs=[arg1_model.sum_pooled_h, arg1_model.all_sum_pooled_h.shape])
    #print f(np.array([2]))

    #f = theano.function(inputs=arg2_model.input, 
           #outputs=[arg2_model.max_pooled_h, arg2_model.all_max_pooled_h.shape])
    #print f(np.array([2]))
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
    output_layer = LinearLayer(rng, 
            n_in_list=n_in_list,
            n_out=num_output_units,
            use_sparse=False,
            X_list=X_list,
            Y=T.lvector(),
            activation_fn=None if use_hinge else T.nnet.softmax)

    nn = NeuralNet()
    layers = [arg1_model, arg2_model] + hidden_layers +  [output_layer] 
    nn.params.extend(arg1_model.params)
    nn.params.extend(arg2_model.params)
    nn.params.extend(output_layer.params)
    for hidden_layer in hidden_layers:
        nn.params.extend(hidden_layer.params)
    nn.layers = layers

    nn.input.extend(arg1_model.input)
    nn.output.extend(output_layer.output)
    nn.predict = output_layer.predict
    nn.hinge_loss = output_layer.hinge_loss
    nn.crossentropy = output_layer.crossentropy
    return nn

