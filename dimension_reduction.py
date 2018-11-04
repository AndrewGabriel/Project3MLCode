#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 18:53:31 2018

@author: andrewgabriel
"""

import os
import sys
import json
import pandas
import numpy
import sklearn
import sklearn.metrics
import sklearn.random_projection
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.mixture
import sklearn.cluster
pandas.set_option('display.max_columns', 500)
import six


from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

import scipy
import scipy.stats
ttest = scipy.stats.ttest_ind


def analyze_single_ttest(ttest_result_list):
    num_classes = len(ttest_result_list)
    bon_correction = (num_classes*(num_classes-1))/2
    
    statistics = []
    if bon_correction == 0.:
        bon_correction=1.
    for i in range(0,num_classes-1):
        for j in range(i+1,num_classes):
            statistics.append(ttest_result_list[i][j])
    
    statistics = pandas.Series(statistics)
    
    p_value = 0.05/bon_correction
    return_dict = {}
    return_dict['p_value'] = p_value
    return_dict['all_signif'] = (1-(1-statistics).prod()) < p_value
    return_dict['test_statistic'] = (1-(1-statistics).prod())
    return_dict['p_values'] = statistics.tolist()
    
    return return_dict


def analyze_component_ttests(ttest_result_dict):
    result_dict = {}
    for key in ttest_result_dict:
        result_dict[key] = analyze_single_ttest(ttest_result_dict[key])
    return result_dict


def means_ttest(components,class_dict):
    p_value_dict    = {}
    num_rows,num_components = components.shape
    
    for component_number in range(num_components):
        
        component = components[:,component_number]
        sub_components = {}
        
        for key in class_dict:
            sub_components[key] = component[class_dict[key]]
            
        all_classes = sorted(list(sub_components.keys()))
        
        p_value_table = [ [1.]*len(all_classes) for i in range(len(all_classes)) ]
        
        
        
        for i in range(0,len(all_classes)-1):
            for j in range(1,len(all_classes)):
                class_i,class_j = all_classes[i],all_classes[j]
                
                stat = ttest(sub_components[class_i],sub_components[class_j],equal_var=False).pvalue
                p_value_table[i][j] = stat
                p_value_table[j][i] = stat
                
        p_value_dict[component_number] = p_value_table
    return p_value_dict


def pca_experiment(X,y,dataset,num_components=25):
    pca = sklearn.decomposition.PCA()
    pca.fit(X)
    
    
    # get the class indexes
    num_classes = len(y.value_counts())
    
    class_inds_bools = {}
    for class_value in y.value_counts().index:
        class_inds_bools[class_value] = y==class_value
    
    
    p_value_dict    = {} # how well each component splits the data against each class
    
    
    # show histogram of eigenvalues
    eigen_values    = pandas.Series(pca.explained_variance_)
    plt.figure()
    eigen_values.hist()
    plt.title("EigenValue Distribution of PCA EigenVectors")
    plt.show()
    
    # compute relevance of each component via two sample ttest
    # on the class distribution of each variable 
    
    
    
    transformed_space = pca.transform(X)
    for component_number in range(transformed_space.shape[1]):
        component = transformed_space[:,component_number]
        sub_components = {}
        
        for class_value in class_inds_bools:
            sub_components[class_value] = component[class_inds_bools[class_value]]
        
        all_classes = sorted(list(sub_components.keys()))
        
        p_value_table = [ [1.]*len(all_classes) for i in range(len(all_classes)) ]
        
        
        
        for i in range(0,len(all_classes)-1):
            for j in range(1,len(all_classes)):
                class_i,class_j = all_classes[i],all_classes[j]
                
                stat = ttest(sub_components[class_i],sub_components[class_j],equal_var=False).pvalue
                p_value_table[i][j] = stat
                p_value_table[j][i] = stat
                
        p_value_dict[component_number] = p_value_table
    
    p_value_analysis = analyze_component_ttests(p_value_dict)
    
    # compute reconstruction error
    
    reconstruction_errors = []
    for k in range(5,num_components+1):
        inverse_components = pca.components_[:k,:]
        transformed_data = pca.transform(X)[:,:k]
        reconstructed_data = numpy.dot(transformed_data,inverse_components)
        reconstruction_loss = ((X-reconstructed_data)**2).mean()
        reconstruction_errors.append([k,reconstruction_loss])
        
    reconstruction_errors = pandas.DataFrame(reconstruction_errors,columns=["K","Recon_Error"])
    reconstruction_errors = reconstruction_errors.set_index("K")
    return p_value_dict,p_value_analysis,reconstruction_errors
        

def normal_guass_kurtosis_values(num_samples,num_sims = 1000):
    
    kurt_values = []
    for i in range(num_sims):
        kurt_value = scipy.stats.kurtosis([numpy.random.normal() for i in range(num_samples)])
        kurt_values.append(kurt_value)
        
    classes = [0 for x in range(num_sims)]
    
    return pandas.DataFrame({'class':classes,'kurtosis':kurt_values})
    

def ica_experiment(X,y,dataset,num_components=25):
    
    ica = sklearn.decomposition.FastICA()
    ica.fit(X)
    components = ica.transform(X)
        
    class_inds_bools = {}
    for class_value in y.value_counts().index:
        class_inds_bools[class_value] = y==class_value
        
        
    ttest_results = means_ttest(components,class_inds_bools)
    
    kurtosis_values = scipy.stats.kurtosis(components)
    kurtosis_classes = [100 for x in range(len(kurtosis_values))]
    
    ica_kurt_df = pandas.DataFrame({'class':kurtosis_classes,'kurtosis':kurtosis_values})
    gauss_kurt_df = normal_guass_kurtosis_values(len(X))
    
    #all_kurts = pandas.concat([ica_kurt_df,gauss_kurt_df],axis=0)
    plt.figure()
    #pandas.plotting.scatter_matrix(all_kurts)
    plt.hist(gauss_kurt_df['kurtosis'])
    plt.title("Guassian Kurtosis distribution of Same Length Vectors")
    plt.show()
    plt.close()
    
    plt.figure()
    plt.hist(ica_kurt_df['kurtosis'])
    plt.title("ICA Kurtosis Distribution of ICA Transformed Data Vectors")
    plt.show()
    plt.close()
    
    plt.figure()
    plt.hist(scipy.stats.kurtosis(X))
    plt.title("%s Kurtosis of variables in original data vectors" % (dataset))
    plt.show()
    plt.close()
    
    p_value_analysis = analyze_component_ttests(ttest_results)
    
    reconstruction_errors = []
    
    for k in range(5,num_components+1):
        ica_kurt_inds= ica_kurt_df.sort_values("kurtosis",ascending=False).index[:k]
    
        mixing = ica.mixing_[:,ica_kurt_inds]
        recon = ica.components_[ica_kurt_inds,:]
        # reconstruct on top n components kurtosis
        
        reconstructed_matrix = numpy.dot(numpy.dot(X,mixing),recon)
        error_value = ((X-reconstructed_matrix)**2).mean()
        reconstruction_errors.append([k,error_value])
        print(error_value)
    
    reconstruction_errors = pandas.DataFrame(reconstruction_errors,columns=["K","Recon_Error"])
    reconstruction_errors = reconstruction_errors.set_index("K")
    return ica,ttest_results,p_value_analysis,reconstruction_errors
    
    
    
def random_projections_experiment(X,y,dataset,num_components,num_simulations = 100):
    # do guassian random projections result in diffentiable 
    # target distributions via ttest
    
    # save the transformed components as well for training later
    
    ttest_results = []
    ttest_analysis_list = []
    transforming_matricies = []
    transformers = []
    
    recon_errors = []
    
    for i in range(num_simulations):
        transformer = sklearn.random_projection.GaussianRandomProjection(num_components)
        all_components = transformer.fit_transform(X)

        transforming_matricies.append(transformer.components_.transpose().copy())
        transformers.append(transformer)
        class_inds_bools = {}
        for class_value in y.value_counts().index:
            class_inds_bools[class_value] = y==class_value
            
        ttest_results.append(means_ttest(all_components,class_inds_bools))    
        ttest_analysis_list.append(analyze_component_ttests(ttest_results[-1]))
        
        recon_data = numpy.dot(all_components,transformer.components_)
        error_value = ((X-recon_data)**2).mean()
        recon_errors.append(error_value)
        
    recon_error = pandas.Series(recon_errors)
    recon_error.hist()
        
        
    return transformers,ttest_analysis_list,recon_error

def factor_analysis_experiment(X,y,dataset,num_components):
    fa = sklearn.decomposition.FactorAnalysis(num_components)
    components = fa.fit_transform(X)
    
    class_inds_bools = {}
    for class_value in y.value_counts().index:
        class_inds_bools[class_value] = y==class_value
    
    recon_errors = []
    for k in range(5,num_components+1):
        sub_components = fa.components_[:k,:]
        recon_data = numpy.dot(fa.transform(X)[:,:k],sub_components)
        error_value = ((X-recon_data)**2).mean()
        recon_errors.append([k,error_value])
    
    recon_errors = pandas.DataFrame(recon_errors,columns=["K","Recon_Error"])
    recon_errors = recon_errors.set_index("K")
    print(recon_errors)
    ttest_results = means_ttest(components,class_inds_bools)
    ttest_analysis = analyze_component_ttests(ttest_results)
    return fa,ttest_analysis,recon_errors


def dimreduction_cluster_experiments(data,dataset_name,kmeans_k,gmm_k=25):
    
    # sklearn.metrics.homogeneity_completeness_v_measure
    
    classes= data['classes']
    features = data['features']
    
    num_components = 10 # for every algorithm
    pca_comps = sklearn.decomposition.PCA(num_components).fit_transform(features)
    
    ica = sklearn.decomposition.FastICA()
    ica_comps = ica.fit_transform(features)
    kurtosis_values = pandas.Series(scipy.stats.kurtosis(ica_comps)).sort_values(ascending=False)
    print(kurtosis_values)
    ica_comps = ica_comps[:,kurtosis_values.index[:num_components]]
    
    rp_comp_list = []
    for i in range(5):
        rp_comp_list.append(sklearn.random_projection.GaussianRandomProjection(num_components).fit_transform(features))
    
    fa_comps = sklearn.decomposition.FactorAnalysis(num_components).fit_transform(features)
    
    
    
    results = {}
    results['KM']= {}
    # compute the measures
    
    kmeans_clustering = sklearn.cluster.KMeans(kmeans_k).fit(features)
    kmeans_clustering = kmeans_clustering.labels_.tolist()
    
    
    comp_names = ['PCA','ICA','RP','FA']
    comp_vals  = [pca_comps,ica_comps,rp_comp_list,fa_comps]
    
    comp_tuples = zip(comp_names,comp_vals)
    
    for name_val,comps in comp_tuples:
        print(name_val)
        if name_val == "RP":
            results["KM"][name_val]= []
            for i in range(len(comps)):
                kmeans_comp = sklearn.cluster.KMeans(kmeans_k)
                comp_val = comps[i]
                kmeans_comp.fit(comp_val)
                labels = kmeans_comp.labels_.tolist()
                stats = sklearn.metrics.homogeneity_completeness_v_measure(kmeans_clustering,labels)
                results['KM'][name_val].append(stats)
        else:
            kmeans_comp = sklearn.cluster.KMeans(kmeans_k)
            kmeans_comp.fit(comps)
            labels = kmeans_comp.labels_.tolist()
            stat = sklearn.metrics.homogeneity_completeness_v_measure(kmeans_clustering,labels)
            results['KM'][name_val] = stat
            
    print(results)
    
    # run the same analysis for GMM, compute probabiltiy distribution
    # of max proba for each to make sure clusterings are very close
    
    results['GMM']= {}
    gmm_clustering = sklearn.mixture.GaussianMixture(gmm_k).fit(features)
    gmm_clustering = gmm_clustering.predict(features).tolist()
    
    gmm_probs = {}
    
    comp_tuples = zip(comp_names,comp_vals)
    for name_val,comps in comp_tuples:
        print(name_val)
        if name_val == "RP":
            results["GMM"][name_val]= []
            gmm_probs[name_val] = []
            for i in range(len(comps)):
                
                gmm_comp = sklearn.mixture.GMM(gmm_k)
                
                comp_val = comps[i]
                gmm_comp.fit(comp_val)
                labels = gmm_comp.predict(comp_val).tolist()
                
                probs = gmm_comp.predict_proba(comp_val)
                top_probs = pandas.Series(numpy.max(probs,axis=1))
                gmm_probs[name_val].append(scipy.stats.describe(top_probs))
                
                stats = sklearn.metrics.homogeneity_completeness_v_measure(gmm_clustering,labels)
                results['GMM'][name_val].append(stats)
        else:
            
            gmm_comp = sklearn.mixture.GMM(gmm_k)
            gmm_comp.fit(comps)
            labels = gmm_comp.predict(comps).tolist()
            probs = gmm_comp.predict_proba(comps)
            top_probs = pandas.Series(numpy.max(probs,axis=1))
            gmm_probs[name_val] = scipy.stats.describe(top_probs)
            
            stat = sklearn.metrics.homogeneity_completeness_v_measure(gmm_clustering,labels)
            results['GMM'][name_val] = stat
    

    print(results['GMM'])
    


    return results,gmm_probs
    
def render_mpl_table(data, col_width=1.5, row_height=0.125*2, font_size=10,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax   
    

def make_plots_dimred_cluster(results,gmm_probs):
    results  =copy.deepcopy(results)
    gmm_probs = copy.deepcopy(gmm_probs)
    gmm_probs['RP'] = gmm_probs['RP'][0] # just take first as example 
    
    
    
    prob_df = pandas.DataFrame(gmm_probs).transpose()
    
    
    cols = ['Num_Samples','Min_Max','Mean','Variance','Skew','Kurt']
    prob_df.columns = cols
    for colname in prob_df:
        if colname == 'Num_Samples':
            continue
        elif colname == 'Min_Max':
            prob_df[colname] = prob_df[colname].map(lambda x: ( round(float(x[0]),3),round(float(x[1]),3) ) )
        else:
            prob_df[colname] = round(prob_df[colname].map(float),3)
    prob_df['Algorithm'] = prob_df.index
    render_mpl_table(prob_df[['Algorithm','Min_Max','Mean','Variance']])
    
    new_cols = ['Homogeneity','Completeness','V_Measure']
    
    
    
    for key in results:
        print(key)
        for algo in results[key]:
            if algo == 'RP':
                data = results[key][algo]
                data = [list(x) for x in data]
                data = [ sum([x[i] for x in data]) / len(data) for i in range(len(data[0])) ]
            else:
                data = results[key][algo]
            data = list(data)
            results[key][algo] = {new_cols[i]:data[i] for i in range(len(new_cols))}
    
        results[key] = pandas.DataFrame(results[key])
        results[key] = round(results[key],3)
        results[key]['Metric'] = results[key].index
        render_mpl_table(results[key])

                    
        
    
    
    
    



def clustering_nn_experiments(mit_data):
    mit_features,mit_targets = mit_data['Train']['features'],mit_data['Train']['classes']
    
    
    kmeans_k = 10
    gmm_k = 25
    
    kmeans_item = sklearn.cluster.KMeans(kmeans_k)
    gmm_item = sklearn.mixture.GaussianMixture(gmm_k)
    
    print("Training KMeans")
    kmeans_item.fit(mit_features)
    print("Training GMM")
    gmm_item.fit(mit_features)
    
    kmeans_vars = pandas.get_dummies(kmeans_item.labels_)
    kmeans_colnames = ["KMeans_Cluster_%i" % (i+1) for i in range(kmeans_vars.shape[1])]
    kmeans_vars.columns = kmeans_colnames
    kmeans_features = kmeans_vars
    #kmeans_features = pandas.DataFrame(mit_features).join(kmeans_vars)
    
    gmm_vars = gmm_item.predict_proba(mit_features)
    gmm_vars = pandas.DataFrame(gmm_vars)
    gmm_vars.columns = ['GMM_Prob_%i' % (i) for i in range(gmm_vars.shape[1])]
    gmm_features = gmm_vars
    #gmm_features = pandas.DataFrame(mit_features).join(gmm_vars)
    
    kmeans_params = {'num_layers':5,'dropout':0.1,'epochs':50,'lr':0.05}
    gmm_params = {'num_layers':5,'dropout':0.2,'epochs':50,'lr':0.05}
    
    #nn_experiment(params,x_train,y_train,dataset_name)
    
    nn_experiment(kmeans_params,kmeans_features,mit_targets,'MIT_KMEANS')
    outfilename = 'Model_Analysis/MIT_KMEANS/kmeans.pkl'
    pickle_object(kmeans_item,outfilename)
    
    nn_experiment(gmm_params,gmm_features,mit_targets,'MIT_GMM')
    outfilename = 'Model_Analysis/MIT_GMM/gmm.pkl'
    pickle_object(gmm_item,outfilename)
    
    
    
    
    
 