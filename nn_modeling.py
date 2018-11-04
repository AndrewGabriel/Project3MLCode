#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 19:01:16 2018

@author: andrewgabriel
"""


import os
import sys
import numpy
import pandas
import json
import sklearn
import copy
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.optimizers
import uuid
import tensorflow 
import tensorflow.estimator
import pickle


def compute_metrics(train_preds,train_targets,test_preds,test_targets,metric_stats={}):
    metric_names = ['f1_score','accuracy','kappa','MCC']
    metrics     = [sklearn.metrics.f1_score,sklearn.metrics.accuracy_score,
                       sklearn.metrics.cohen_kappa_score,sklearn.metrics.matthews_corrcoef]

    metric_iterator     = zip(metric_names,metrics)
    is_multiclass = len(train_targets.value_counts()) > 2

    for metric_name,metric_function in metric_iterator:
        #
        if metric_name is 'f1_score' and is_multiclass:
            metric_stats['IS_'+metric_name] = metric_function(train_targets,train_preds,average='macro')
            metric_stats['OOS_'+metric_name] = metric_function(test_targets,test_preds,average='macro')
            continue
        metric_stats['IS_'+metric_name] = metric_function(train_targets,train_preds)
        metric_stats['OOS_'+metric_name] = metric_function(test_targets,test_preds)

    return metric_stats


def nn_experiment(params,x_train,y_train,dataset_name):
    
    # Dataset name should take into account the data and algorithm used to 
    # decompose
    
    model_out_dir = 'Model_Analysis/%s/' % (dataset_name)
    try:
        os.mkdir(model_out_dir)
    except:
        pass
    
    x_train = x_train.copy()
    y_train = y_train.copy()
    
    params = copy.deepcopy(params)
    is_binary       = False
    num_classes     = len(y_train.value_counts())

    if num_classes == 2:
        is_binary = True
        class_weights = float(len(y_train)) / y_train.value_counts() # already 0-1
        activation = 'sigmoid'
    else:
        class_weights = float(len(y_train)) / y_train.value_counts() 
        class_weights = json.loads(class_weights.to_json())
        cols = sorted(y_train.value_counts().index.tolist())
        class_weights = {i:class_weights[str(cols[i])] for i in range(len(cols))}
        y_train         = pandas.get_dummies(y_train)
        activation = 'softmax'

    num_layers      = params['num_layers']
    dropout_rate    = params['dropout']
    num_epochs      = params['epochs']
    lr              = params['lr']
    params['uuid']  = str(uuid.uuid4())
    
    
    layer_size = 2**num_layers
    model   = Sequential()
    model.add( Dense(layer_size,input_shape=(x_train.shape[1],),activation=activation) )
    layer_size      = layer_size // 2

    if is_binary:
        num_classes = 1
        
    while layer_size > num_classes: # 1 for binary, 3+ for multiple classes
        model.add(Dense(layer_size,activation=activation))
        if dropout_rate>0.:
            model.add(Dropout(dropout_rate))
        layer_size = layer_size // 2

    model.add(Dense(num_classes))
    model.add(Activation(activation))
    
    opt = keras.optimizers.Adam(lr=lr,clipnorm=1.,clipvalue=1.)
    
    if is_binary:
        model.compile(loss='binary_crossentropy',optimizer=opt)
    else:
        model.compile(loss='categorical_crossentropy',optimizer=opt)
        
    for layer in model.layers:
        print(layer.input_shape)
        
    outfilename = model_out_dir + 'model.json'
    model_json = json.loads(model.to_json())
    with open(outfilename,'w') as file_handle:
        json.dump(model_json,file_handle)
        
    outfilename = model_out_dir + 'params.json'
    
    with open(outfilename,'w') as file_handle:
        json.dump(params,file_handle)
    
    
    out_dir = model_out_dir + 'NN_Weights/'
    try:
        os.mkdir(out_dir)
    except:
        pass
    
    for epoch_num in range(num_epochs):
        model.fit(x_train,y_train,epochs=1,verbose=True,class_weight=class_weights)              
        model_weights   = model.get_weights()
        outfilename     = out_dir + str(epoch_num) + '.model_weights'
        outfile         = open(outfilename,'w')
        model_weights   = list(map(lambda x: x.tolist(),model_weights))
        json.dump(model_weights,outfile,indent=4,sort_keys=True)
        outfile.close()


def pickle_object(item,outfilename):
    with open(outfilename,'wb') as file_handle:
        pickle.dump(item,file_handle)
    
    
    
def generate_f1_learning_curve(model,weights_dir,x_train,y_train,x_test,y_test):
    
    
    
    weight_files = os.listdir(weights_dir)
    weight_files = sorted(weight_files,key= lambda x: int(x.split('.')[0]))
    weight_files = [weights_dir+filename for filename in weight_files]
    
    
    learning_curve = [] # list of dicts of in-sample and OOS F1-scores
    for filename in weight_files:
        weights = json.load(open(filename))
        model.set_weights(weights)
        
        train_preds = model.predict(x_train)
        test_preds = model.predict(x_test)

        train_preds = numpy.argmax(train_preds,axis=1).flatten().tolist()
        test_preds = numpy.argmax(test_preds,axis=1).flatten().tolist()
        metrics = {}
        compute_metrics(train_preds,y_train,test_preds,y_test,metrics)
        learning_curve.append(metrics)
    
    learning_curve = pandas.DataFrame(learning_curve)
    print(learning_curve)
    return learning_curve


def process_in_dir(in_dir):
    return_dict = {}
    return_dict['weights_dir'] = in_dir + 'NN_Weights/'
    model = json.load(open(in_dir+'model.json'))
    model = json.dumps(model)
    model = keras.models.model_from_json(model)
    opt = keras.optimizers.Adam(lr=0.1,clipnorm=1.,clipvalue=1.)
    model.compile(loss='categorical_crossentropy',optimizer=opt)
    return_dict['model'] = model
    
    pickle_file = [x for x in os.listdir(in_dir) if x.endswith('.pkl')][0]
    
    return_dict['preprocesser'] = pickle.load(open(in_dir+pickle_file,'rb'))
    return_dict['in_dir'] = in_dir
    return return_dict
    

def generate_all_learning_curves(mit_data,in_dir = 'Model_Analysis/'):
    
    
    
    decomp_dirs = ['MIT_FA/', 'MIT_ICA/', 'MIT_PCA/']
    decomp_dirs= [in_dir + x for x in decomp_dirs]
    
    cluster_dirs = ['MIT_KMEANS/','MIT_GMM/']
    cluster_dirs = [in_dir + x for x in cluster_dirs]
    
    rp_dir = 'MIT_Random_Projections/'
    rp_dirs = [in_dir + rp_dir + x + '/' for x in os.listdir(in_dir+rp_dir)]
    
    learning_curves = {}
    '''
    for decomp_dir in decomp_dirs:
        print(decomp_dir)
        dict_val = process_in_dir(decomp_dir)
        x_train = dict_val['preprocesser'].transform(mit_data['Train']['features'])[:,:25]
        x_test =  dict_val['preprocesser'].transform(mit_data['Test']['features'])[:,:25]
        y_train,y_test = mit_data['Train']['classes'],mit_data['Test']['classes']
        
        model = dict_val['model']
        weights_dir = dict_val['weights_dir']
        
        learning_curves[decomp_dir.strip('/').split('/')[-1]] = generate_f1_learning_curve(model,weights_dir,x_train,y_train,x_test,y_test)
    
    learning_curves['MIT_RP'] = []
    for rp_dir in rp_dirs:
        print(rp_dir)
        dict_val = process_in_dir(rp_dir)
        x_train = dict_val['preprocesser'].transform(mit_data['Train']['features'])[:,:25]
        x_test =  dict_val['preprocesser'].transform(mit_data['Test']['features'])[:,:25]
        y_train,y_test = mit_data['Train']['classes'],mit_data['Test']['classes']
        
        model = dict_val['model']
        weights_dir = dict_val['weights_dir']
        
        lr = generate_f1_learning_curve(model,weights_dir,x_train,y_train,x_test,y_test)
        learning_curves['MIT_RP'].append(lr)
        
    '''
    
    # do KMEANS 
    print("MIT_KMEANS")
    x_train,x_test = mit_data['Train']['features'] , mit_data['Test']['features']
    y_train,y_test = mit_data['Train']['classes']  , mit_data['Test']['classes']
    
    
    dict_val = process_in_dir(cluster_dirs[0])
    model = dict_val['model']
    
    kmeans_x_train  = pandas.get_dummies(dict_val['preprocesser'].predict(x_train))
    kmeans_x_test   = pandas.get_dummies(dict_val['preprocesser'].predict(x_test))
    
    #kmeans_x_train = pandas.DataFrame(x_train).join(train_clusters,rsuffix='KM')
    #kmeans_x_test = pandas.DataFrame(x_test).join(test_clusters,rsuffix='KM')
    print(kmeans_x_train)
    print(kmeans_x_test)
    weights_dir = dict_val['weights_dir']
    
    learning_curves['MIT_KMEANS'] = generate_f1_learning_curve(model,weights_dir,kmeans_x_train,y_train,kmeans_x_test,y_test)
    
        
    # do GMM
    
    print("MIT_GMM")
    dict_val = process_in_dir(cluster_dirs[1])
    model = dict_val['model']
    
    gmm_x_train  = pandas.DataFrame(dict_val['preprocesser'].predict_proba(x_train))
    gmm_x_test   = pandas.DataFrame(dict_val['preprocesser'].predict_proba(x_test))
    
    #gmm_x_train = pandas.DataFrame(x_train).join(train_clusters,rsuffix='GMM')
    print(gmm_x_train)
    #gmm_x_test = pandas.DataFrame(x_test).join(test_clusters,rsuffix='GMM')
    print(gmm_x_test)
    weights_dir = dict_val['weights_dir']
    
    learning_curves['MIT_GMM'] = generate_f1_learning_curve(model,weights_dir,gmm_x_train,y_train,gmm_x_test,y_test)
    
    return learning_curves
    