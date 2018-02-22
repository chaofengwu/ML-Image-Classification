from __future__ import print_function

from resizeimage import resizeimage
from PIL import Image

from time import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import get_file_list

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import cross_validation

from scipy import misc

from numpy import genfromtxt
import pickle

def test(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=42)
	n_components = 30
	pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

	eigenfaces = pca.components_.reshape((n_components, 300, 300))

	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	print(X_test_pca[0])

	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
	              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

	clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	clf = clf.fit(X_train_pca, y_train)

	print(clf.best_estimator_)


	y_pred = clf.predict(X_test_pca)
	count = 0
	for i in range(len(y_test)):
		if y_pred[i] == y_test[i]:
			count += 1
	print(count/len(y_test))
	print(y_pred)
	print(y_test)


def train(X_train, y_train):
	n_components = 30
	pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

	eigenfaces = pca.components_.reshape((n_components, 300, 300))

	X_train_pca = pca.transform(X_train)
	print(X_test_pca[0])

	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
	              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

	clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	clf = clf.fit(X_train_pca, y_train)
	pickle.dump(pca, open('pca_model.sav'))
	pickle.dump(clf, open('clf_model.sav'))


def predict(X):
	try:
		pca = pickle.load(open('pca_model.sav'))
		pca = pickle.load(open('clf_model.sav'))
	except:
		print('please first train the model.')

	X_pca = pca.transform(X)
	y_pred = clf.predict(X_pca)
	