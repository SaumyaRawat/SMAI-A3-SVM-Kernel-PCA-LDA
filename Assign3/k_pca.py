from scipy.spatial.distance import pdist, squareform
from scipy import exp
import numpy as np
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import rbf_kernel

def kpca(X, gamma=15, k=10):
    
    # Calculating the distances for every pair of points in the NxD dimensional dataset.
    _dists = pdist(X, 'minkowski')

    # Converting the pairwise distances into a symmetric NxN matrix.
    sym_dists = squareform(_dists)

    # Computing the NxN kernel matrix.
    K = exp(-gamma * sym_dists)

    #Centering Kernel since data has to be standardizied
    kern_cent = KernelCenterer()
    K = kern_cent.fit_transform(K)
    
    eig_vals, eig_vecs = np.linalg.eig(K)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    vec = np.array([ eig_pairs[i][1] for i in range(k)])
    vec = vec.T # to make eigen vector matrix nxk

    return vec