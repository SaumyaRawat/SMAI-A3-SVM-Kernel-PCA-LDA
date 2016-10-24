#!/bin/python

#import global_variables as gv
#from pca import pca_X
import numpy as np
import pandas as pd
from sklearn.preprocessing import KernelCenterer
import numpy as np
import array
from scipy.sparse import csr_matrix, linalg, lil_matrix
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
from copy import deepcopy

global train_labels1, train_labels2, lda_X1, lda_X2
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
                      
    class_indices = []
    class_indices.append(np.where(train_labels==-1)) #at pos 0
    class_indices.append(np.where(train_labels==1))  #at pos 1
    
    
    
    #dimension
    n = X.shape[0]
    d = X.shape[1]
    
    n1 = X[class_indices[0]].shape[0]
    n2 = X[class_indices[1]].shape[0]
    
    #Calculate Kernel matrix K1 n1xn1
    K1=np.zeros((n,n1))
    class_no = 0
    for i,xi in enumerate(X):
        for j,xj in enumerate(X[class_indices[class_no]]):
            K1[i][j] = (xi.dot(xj.T))
    
    
    #Calculate Kernel matrix K2 n2xn2
    K2=np.zeros((n,n2))
    class_no = 1
    for i,xi in enumerate(X):
        for j,xj in enumerate(X[class_indices[class_no]]):
            K2[i][j] = (xi.dot(xj.T))
    
    
    #Calculate means
    #Calculate mean matrix M1
    M1=np.zeros((n,1))
    class_no = 0
    for i,xi in enumerate(X):
            M1[i] = np.sum([[xi.dot(xj.T)] for xj in X[class_indices[class_no]]])
    
    #Calculate mean matrix M2
    M2=np.zeros((n,1))
    class_no = 1
    for i,xi in enumerate(X):
            M2[i] = np.sum([[xi.dot(xj.T)] for xj in X[class_indices[class_no]]])
    
    #Calculate between class scatter matrix
    M = (M2 - M1).dot((M2 - M1).T)
    
    #Calculate within class scatter matrix
    N1 = (K1.dot(np.identity(n1) - (np.ones(n1)*n1))).dot(K1.T)
    N2 = (K2.dot(np.identity(n2) - (np.ones(n2)*n2))).dot(K2.T)
    N = N1 + N2;
                           
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(N).dot(M))
    
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    
    # Construct KxD eigenvector matrix W
    W = eig_pairs[0][1]
    lda_X = np.real(X.T.dot(W))
    if dataset_index == 0:
        lda_X1 = lda_X
        train_labels1 = train_labels
    else:
        lda_X2 = lda_X
        train_labels2 = train_labels

