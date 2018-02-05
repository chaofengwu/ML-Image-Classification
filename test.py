
from __future__ import print_function

from resizeimage import resizeimage
from PIL import Image

from time import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import csv
import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import cross_validation

from scipy import misc

from numpy import genfromtxt

w,h = 300,300

def read_file_list(file_name):
	with open(file_name) as f:
		content = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
		content = [x.strip() for x in content]
	return content


def get_image(file_name):
	X = []
	file_list = read_file_list(file_name)
	valid_list = []
	idx = 0
	for i in file_list:
		try:
			image = Image.open(i)
			image = image.resize((w,h), Image.ANTIALIAS)
			image = image.convert('L')
			img_array = list(image.getdata())
			X.append(np.array(img_array))
			valid_list.append(idx)
			
		except:
			pass
		idx += 1
	return X, valid_list


def get_label_data(file_name, data_choosed):
	f = open(file_name)
	content = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip()[0] for x in content]
	res = []
	for i in range(len(data_choosed)):
		res.append(int(content[data_choosed[i]]))
	return res

def wtf():
	X, valid_list = get_image('/home/chaofeng/Documents/practicum/file_list.txt')
	y = get_label_data('/home/chaofeng/Documents/practicum/test_label.txt', valid_list)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=42)

	n_components = 30
	pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

	eigenfaces = pca.components_.reshape((n_components, 300, 300))

	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)


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


X, valid_list = get_image('/home/chaofeng/Documents/practicum/file_list.txt')
y = get_label_data('/home/chaofeng/Documents/practicum/test_label.txt', valid_list)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=42)

n_components = 30
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

eigenfaces = pca.components_.reshape((n_components, 300, 300))

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = cross_validation.cross_val_score

print(clf.best_estimator_)


y_pred = clf.predict(X_test_pca)
