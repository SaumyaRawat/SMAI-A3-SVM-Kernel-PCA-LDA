import random
import numpy as np
from lda import lda_X1,lda_X2, train_labels1, train_labels2
from pca import pca_X1,pca_X2#, train_labels1, train_labels2
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import warnings


class Model:
	def __init__(self, X, labels):
		self.dataset = X
		self.class_labels = labels
		self.test_class_labels = list(labels)
		self.train_class_labels = []
		self.trainingSet= []
		self.testSet = []
		self.svc = []


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
		self.svc = svm.SVC(kernel='linear', C=1,gamma='auto').fit(np.array(tr_X).reshape(-1,1), labels)

	def train_1Drbf_classifier(self):
		tr_X = self.trainingSet;
		labels = self.train_class_labels
		#C = 1.0 # SVM regularization parameter
		self.svc = svm.SVC(kernel='rbf', C=1,gamma='auto').fit(np.array(tr_X).reshape(-1,1), labels)

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
		
		# PCA
		model_pca = Model(pca_X1,train_labels1)
		splitRatio = 0.85
		model_pca.k_fold_split(splitRatio)
		print('Size of train=',len(model_pca.trainingSet),' and test=',len(model_pca.testSet))
		model_pca.train_linear_classifier()
		predictions = model_pca.test_classifier()
		accuracy_pca = model_pca.get_accuracy(predictions)
		print('Accuracy after PCA(K==100) and using Linear Kernel: ',accuracy_pca)
		
		# LDA
		model_lda = Model(lda_X1,train_labels1)
		splitRatio = 0.9
		model_lda.k_fold_split(splitRatio)
		print('Size of train=',len(model_lda.trainingSet),' and test=',len(model_lda.testSet))
		model_lda.train_1Drbf_classifier()
		predictions = model_lda.test_classifier()
		accuracy_lda = model_lda.get_accuracy(predictions)
		print('Accuracy after LDA and using RBF Kernel: ',accuracy_lda)