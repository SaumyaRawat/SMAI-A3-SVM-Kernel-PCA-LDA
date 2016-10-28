import numpy as np
import mlpy

np.random.seed(0)
def klda(X,y):
    K = mlpy.kernel_gaussian(X, X, sigma=15)	
    kfda = mlpy.KFDA(lmb=0.0)
    kfda.learn(K, y) # compute the tranformation vector
    z = kfda.transform(K) # embedded x into the kernel fisher space
    return (z)