from sklearn.decomposition import KernelPCA


def kpca(X,g,k):
	kpca = KernelPCA(k,kernel="rbf", fit_inverse_transform=True, gamma=g)     		        		
	pcaX = kpca.fit_transform(X)                
	return (pcaX)