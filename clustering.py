#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:28:53 2018

@author: andrewgabriel
"""

import os
import sys
import json
import pandas
import sklearn
import sklearn.metrics
import sklearn.random_projection
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.mixture
import sklearn.cluster
import numpy
import scipy
import scipy.stats

from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import random
# Params
'''
higgs_kmeans: k = 15 via elbow method
mit_kmeans: k = 10 via elbow method
'''


'''
Silhouette measure
Adjusted Rand index
V measure for completeness
'''

def read_in_data():
    higgs_train,higgs_test  = pandas.read_csv("Train_Data/higgs.csv"),pandas.read_csv("Test_Data/higgs.csv")
    mit_train,mit_test  = pandas.read_csv("Train_Data/mit.csv"),pandas.read_csv("Test_Data/mit.csv")

    return_dict = {}
    return_dict['higgs'] = {} 
    return_dict['mit'] = {}
    
    higgs_train_features,higgs_test_features = higgs_train.drop("CLASS",axis=1),higgs_test.drop("CLASS",axis=1)
    mit_train_features,mit_test_features = mit_train.drop("CLASS",axis=1),mit_test.drop("CLASS",axis=1)

    higgs_train_classes,higgs_test_classes  = higgs_train['CLASS'],higgs_test['CLASS']
    mit_train_classes,mit_test_classes  = mit_train['CLASS'],mit_test['CLASS']
    
    higgs_scaler    = sklearn.preprocessing.RobustScaler()
    higgs_scaler.fit(higgs_train_features)
    
    
    return_dict['higgs']['scaler'] = higgs_scaler
    
    return_dict['higgs']['Train'] = {} 
    return_dict['higgs']['Train']['features'] = higgs_scaler.transform(higgs_train_features)
    return_dict['higgs']['Train']['classes'] = higgs_train_classes
    
    return_dict['higgs']['Test'] = {}
    return_dict['higgs']['Test']['features'] = higgs_scaler.transform(higgs_test_features)
    return_dict['higgs']['Test']['classes'] = higgs_test_classes
    
    mit_scaler = sklearn.preprocessing.RobustScaler()
    mit_scaler.fit(mit_train_features)
    
    return_dict['mit']['scaler'] = mit_scaler
    
    return_dict['mit']['Train'] = {}
    return_dict['mit']['Train']['features'] = mit_scaler.transform(mit_train_features)
    return_dict['mit']['Train']['classes'] = mit_train_classes
    
    return_dict['mit']['Test'] = {}
    return_dict['mit']['Test']['features'] = mit_scaler.transform(mit_test_features)
    return_dict['mit']['Test']['classes'] = mit_test_classes
    
    return return_dict


# GMM already computes AIC
# https://stats.stackexchange.com/questions/85929/corrected-aic-aicc-for-k-means
def kmeans_aic(kmeans,X):
    
    '''
    m = ncol(fit$centers)
    n = length(fit$cluster)
    k = nrow(fit$centers)
    D = fit$tot.withinss
    return(D + 2*m*k)
    '''
    
    
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand

    return kmeans.inertia_ + 2*d*N

    '''
    const_term = 0.5 * m * np.log10(N)

    BIC = np.sum([n[i] * np.log10(n[i]) -
           n[i] * np.log10(N) -
         ((n[i] * d) / 2) * np.log10(2*np.pi) -
          (n[i] / 2) * np.log10(cl_var[i]) -
         ((n[i] - m) / 2) for i in range(m)]) - const_term

    return(BIC)
    '''
    
def kmeans_params(X,k_values = [2,3,4,5,10,15,25],n_simulations=5,max_iter=500):
    
    results     = {}
    for k in k_values:
        print("Computing AIC Values for %i" % (k))
        results[k] = []
        for n in range(n_simulations):
            kmeans = sklearn.cluster.KMeans(k,max_iter=max_iter)
            kmeans = kmeans.fit(X)
            if kmeans.n_iter_ >= max_iter:
                results[k].append(None)
            else:
                results[k].append(kmeans_aic(kmeans,X))
        print(results[k])
      
    for k in results:
        series_value    = pandas.Series(results[k])
        series_value = series_value.replace(np.NaN,series_value.max(skipna=True))
        results[k] = series_value.mean()
        
    return results
    
def gmm_params(X,k_values = [2,3,4,5,10,15,25],n_simulations=1,max_iter=500):
    results = {}
    for k in k_values:
        print("Computing AIC Values for %i" % (k))
        results[k] = []
        for n in range(n_simulations):
            gmm = sklearn.mixture.GaussianMixture(n_components=k)
            gmm = gmm.fit(X)
            if gmm.converged_:
                results[k].append(gmm.aic(X))
            else:
                results[k].append(None)
    
    for k in results:
        series_value    = pandas.Series(results[k])
        series_value = series_value.replace(np.NaN,series_value.max(skipna=True))
        results[k] = series_value.mean()
    
    return results
                    

def kmeans_experiment(X,y,k,dataset,max_iter=500):
    kmeans = sklearn.cluster.KMeans(k,max_iter=max_iter)
    kmeans.fit(X)
    
    
    pred_y = y.copy()
    
    results = {}
    print(kmeans.labels_)
    centers = kmeans.cluster_centers_
    plot_pca = sklearn.decomposition.PCA(3)
    values = plot_pca.fit_transform(centers)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(values[:,0],values[:,1],values[:,2],c='black',s=200,alpha=0.5)
    #plt.scatter(values[:,0],values[:,1],c='black', s=200, alpha=0.5)
    print(values)
    plt.title("PCA Reduced Cluster Centers to three dimensions for %s Dataset" % (dataset))
    plt.show()
    plt.close()
    
    # Do the analysis of the data
    
    # Hist of cluster sizes
    cluster_values  = pandas.Series(kmeans.labels_)
    cluster_sizes = cluster_values.value_counts()
    cluster_sizes = cluster_sizes[sorted(cluster_sizes.index,key=int)]
    cluster_sizes.plot(kind='bar',title='Cluster Sizes for %s Dataset' % (dataset))
    
    # get majority vote for each
    
    cluster_labels     = {}
    cluster_indexes     = {}
    for i in range(len(cluster_values)):
        
        cluster_ind = cluster_values[i]
        
        if cluster_ind not in cluster_labels:
            cluster_indexes[cluster_ind] = []
            cluster_labels[cluster_ind] = []
            
        cluster_labels[cluster_ind].append(y[i])
        cluster_indexes[cluster_ind].append(i)
        
    
    for cluster_ind in cluster_indexes:
        values = pandas.Series(cluster_labels[cluster_ind]).value_counts()
    
        cluster_labels[cluster_ind] = int(np.argmax(values))
        pred_y.ix[cluster_indexes[cluster_ind]] = cluster_labels[cluster_ind]
    
    print(pred_y.value_counts())
    results['dominant_class_distribution'] = json.loads(pred_y.value_counts().to_json())
    results['f1_score'] = sklearn.metrics.f1_score(y,pred_y,average='macro')
    results['MI'] = sklearn.metrics.adjusted_mutual_info_score(y,pred_y)
    results['rand_score'] = sklearn.metrics.adjusted_rand_score(y,cluster_values)
    
    results['homogeniety'] = sklearn.metrics.homogeneity_completeness_v_measure(y,cluster_values)
    
    # sample 5% of data to get silhoutte scores

    sil_scores = []
    for i in range(50):
        sil_scores.append(sklearn.metrics.silhouette_score(X,cluster_values,sample_size = int(0.05*len(cluster_values))))
    sil_scores = pandas.Series(sil_scores)
    plt.figure()
    plt.hist(sil_scores)
    plt.title("Histogram of Silhouette Score Samples for 5 Percent sample data for %s" % (dataset))
    plt.show()
    
    return results
 
def gmm_experiment(X,y,k,dataset,max_iter=500):
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)
    gmm_model.fit(X)
    print(gmm_model.converged_)
    
    centers = gmm_model.means_
    plot_pca = sklearn.decomposition.PCA(3)
    values = plot_pca.fit_transform(centers)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(values[:,0],values[:,1],values[:,2],c='black',s=200,alpha=0.5)
    #plt.scatter(values[:,0],values[:,1],c='black', s=200, alpha=0.5)
    print(values)
    plt.title("PCA Reduced Cluster Centers to three dimensions for %s Dataset" % (dataset))
    plt.show()
    plt.close()
    
    
    spaces = list(map(numpy.linalg.norm,[gmm_model.covariances_[i] for i in range(k)]))
    spaces = pandas.Series(spaces)
    
    plt.figure()
    plt.title("GMM Gaussian Variance (Frobenious Norm) Histogram for dataset %s" % (dataset))
    spaces.hist()
    plt.show()
    plt.close()
    
    scored_data = gmm_model.predict_proba(X)
    class_inds_bools = {}
    for class_value in y.value_counts().index:
        class_inds_bools[class_value] = y==class_value
        
    ttest_result = means_ttest(scored_data,class_inds_bools)
    
    
    # how close to each cluster?
    
    top_probs = pandas.Series(numpy.max(scored_data,axis=1))
    plt.figure()
    top_probs.hist()
    plt.title("Highest Probability of Assignment Histogram for %s Dataset" % (dataset))
    plt.show()
    plt.close()
    print(scipy.stats.describe(top_probs))
    
    
    return gmm_model
    
    
