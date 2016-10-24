#!/bin/python

import pandas as pd
from sklearn.preprocessing import KernelCenterer
import numpy as np
from scipy.sparse import lil_matrix
from copy import deepcopy

global pca_X1,pca_X2,train_labels1,train_labels2

for dataset_index in range(2):
    if dataset_index == 0:
        train_labels=np.loadtxt('dataset/arcene_train.labels')
        lines = open('dataset/arcene_train.data', 'r').readlines()
        dataList = [line.rstrip('\n') for line in lines]
        k = len(max(dataList,key=len))
        cols = np.arange(k)
        
        train_data = pd.read_csv(
            filepath_or_buffer='dataset/arcene_train.data', 
            header=None,
            sep=" ",
            names=cols,
            engine = 'python')
        train_data=train_data.fillna(0)
        X = ((train_data.values))

    elif dataset_index == 1:
        train_labels=np.loadtxt('dataset/dexter_train.labels')
        lines = open('dataset/dexter_train.data', 'r').readlines()
        dataList = [line.rstrip('\n') for line in lines]
        n = len(dataList)
        X = np.zeros((n,20000))
        i =0
        for line in (dataList):
            line_ele = line.split()
            for element in line_ele:
                col = element.split(':')[0]
                attr_value = element.split(':')[1]
                X[i][int(col)] = attr_value
            i = i+1
   
    
    train_data=train_data.fillna(0)
    X = lil_matrix((train_data.values))
    
    k=100
    
    #Kernel_PCA since d>>n
    K=X.dot(X.T)
    
    #Centering Kernel since data has to be standardizied
    kern_cent = KernelCenterer()
    S = kern_cent.fit_transform(K.toarray())
    
    #val,vec=linalg.eigs(S,k,which='LM')
    
    eig_vals, eig_vecs = np.linalg.eig(S)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    vec = np.array([ eig_pairs[i][1] for i in range(k)])
    vec = vec.T # to make eigen vector matrix nxk
    
    # d×k-dimensional eigenvector matrix W.
    W=X.T.dot(vec)
    Y=X.dot(W)
    
    pca_X = deepcopy(Y)
    if dataset_index == 0:
        pca_X1 = pca_X
        train_labels1 = train_labels
    elif dataset_index == 1:
        pca_X2 = pca_X
        train_labels2 = train_labels
        