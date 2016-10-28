#!/bin/python
import mlpy
import numpy as np

def klda(X,y):                      
    class_indices = []
    class_indices.append(np.where(y==-1)) #at pos 0
    class_indices.append(np.where(y==1))  #at pos 1
   
    #dimension
    n = X.shape[0]
    d = X.shape[1]
    
    n1 = X[class_indices[0]].shape[0]
    n2 = X[class_indices[1]].shape[0]
    K = X.dot(X.T)
    
#Calculate Kernel matrix K1 n1xn1									
    K1 = mlpy.kernel_gaussian(X, X[class_indices[0]], sigma=15)										
    K2 = mlpy.kernel_gaussian(X, X[class_indices[1]], sigma=15)										
    
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
    ldaX = []
    for j in range(len(X)):
     	ldaX.append( sum([W[i] * mlpy.kernel_gaussian(X[j], X[i], sigma=15) for i in range(n) ]))
    return ldaX

