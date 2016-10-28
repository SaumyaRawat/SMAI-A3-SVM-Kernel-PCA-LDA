import random
#import numpy as np
#from lda import lda_X1,lda_X2, train_labels1, train_labels2
#from k_pca import pca_X1,pca_X2#, train_labels1, train_labels2
from sklearn import svm
#import warnings
#from plot_kernel_pca import X_pca
#from sklearn.decomposition import PCA, KernelPCA
from copy import deepcopy
import k_pca
import k_lda
import import_database
import ib_klda, ib_kpca

def cross_validation(X_pca, y, test_size=0.4, random_state=0):
	trainSize = int(len(X_pca) * test_size)
	X_train = []
	y_test = list(y)
	y_train = []
	copy = list(X_pca)
	while len(X_train) < trainSize:
		index = random.randrange(len(copy))
		X_train.append(copy.pop(index))
		y_train.append(y[index])
		y_test.pop(index)
	X_test = deepcopy(copy)
	return(X_train, X_test, y_train, y_test)
 
class Model:
	def __init__(self, X, labels):
		self.dataset = X
		self.class_labels = labels
		self.test_class_labels = list(labels)
		self.train_class_labels = []
		self.trainingSet= []
		self.testSet = []
		self.svc = svm.SVC()


	def k_fold_split(self,splitRatio):
		trainSize = int(len(self.dataset) * splitRatio)
		trainSet = []
		copy = list(self.dataset)
		while len(trainSet) < trainSize:
			index = random.randrange(len(copy))
			trainSet.append(copy.pop(index))
			self.train_class_labels.append(self.class_labels[index])
			self.test_class_labels.pop(index)
		self.trainingSet = trainSet
		self.testSet = copy
	
	def train_linear_classifier(self):
		tr_X = self.trainingSet;
		labels = self.train_class_labels
		#C = 1.0 # SVM regularization parameter
		self.svc = svm.SVC(kernel='linear', C=1,gamma='auto').fit(tr_X, labels)

	def train_rbf_classifier(self):
		tr_X = self.trainingSet;
		labels = self.train_class_labels
		#C = 1.0 # SVM regularization parameter
		self.svc = svm.SVC(kernel='rbf', C=1,gamma='auto').fit(tr_X, labels)
	
	def train_1Dlinear_classifier(self):
		tr_X = self.trainingSet;
		labels = self.train_class_labels
		#C = 1.0 # SVM regularization parameter
		self.svc = svm.SVC(kernel='linear', C=1,gamma='auto').fit(tr_X, labels)

	def train_1Drbf_classifier(self):
		tr_X = self.trainingSet;
		labels = self.train_class_labels
		#C = 1.0 # SVM regularization parameter
		self.svc = svm.SVC(kernel='rbf', C=1,gamma='auto').fit(tr_X, labels)

	def predict(self, inputVector):
		bestLabel = self.svc.predict([inputVector])
		return bestLabel
	
	def test_classifier(self):
		predictions = []
		for i in range(len(self.testSet)):
			result = self.predict(self.testSet[i])
			predictions.append(result)
		return predictions
	
	def get_accuracy(self,predictions):
		correct = 0
		for i in range(len(self.testSet)):
			if self.test_class_labels[i] == predictions[i][0]:
				correct += 1
		return (correct/float(len(self.testSet))) * 100.0

if __name__ == "__main__":    
		#warnings.simplefilter("error")
		X,y = import_database.import_database('arcene')  
##############################################################################################
		myLdaX = k_lda.klda(X,y)
		arcene_klda = Model(ldaX,y)
		arcene_klda.k_fold_split(0.9)
		print('Size of train=',len(arcene_klda.trainingSet),' and test=',len(arcene_klda.testSet))
		
		print('LDA, Linear SVM on Arcene Dataset: ')
		arcene_klda.train_1Dlinear_classifier()
		predictions_lda1 = arcene_klda.test_classifier()
		accuracy_lda1 = arcene_klda.get_accuracy(predictions_lda1)
		print('Accuracy after LDA and using Linear SVM Kernel: ',accuracy_lda1,'\n')
	

		print('LDA, RBF SVM on Arcene Dataset: ')
		arcene_klda.train_1Drbf_classifier()
		predictions_lda2 = arcene_klda.test_classifier()
		accuracy_lda2 = arcene_klda.get_accuracy(predictions_lda2)
		print('Accuracy after LDA and using Linear SVM Kernel: ',accuracy_lda2,'\n')
	
##############################################################################################

		k = 100		
		pcaX = k_pca.kpca(X,15,k)
		#inbuilt_pca = ib_kpca.kpca(X,15,k)		
		arcene_kpca = Model(pcaX,y)
		arcene_kpca.k_fold_split(0.9)
		print('Size of train=',len(arcene_kpca.trainingSet),' and test=',len(arcene_kpca.testSet))
		
		print('PCA, Linear SVM on Arcene Dataset: ')
		arcene_kpca.train_linear_classifier()
		predictions1 = arcene_kpca.test_classifier()
		accuracy_pca1 = arcene_kpca.get_accuracy(predictions1)#arcene_kpca.svc.score(arcene_kpca.testSet,arcene_kpca.test_class_labels)
		print('Accuracy after PCA(K==',k,') and using Linear SVM Kernel: ',accuracy_pca1,'\n')
	
		print('PCA, RBF SVM on Arcene Dataset: ')
		arcene_kpca.train_rbf_classifier()
		predictions2 = arcene_kpca.test_classifier()
		accuracy_pca2 = arcene_kpca.get_accuracy(predictions2)#arcene_kpca.svc.score(arcene_kpca.testSet,arcene_kpca.test_class_labels)
		print('Accuracy after PCA(K==',k,') and using RBF SVM Kernel: ',accuracy_pca2,'\n')

