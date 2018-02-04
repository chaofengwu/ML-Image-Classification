from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import numpy as np
# Input data:
# 	color 5
# 	histo 3
# 	mean 3
# 	image_pca 10

########### Train

##### Get input data
def get_train_data(color, histo, pca_ima, square):
	data_size = len(color)
	feature_data = []
	for i in range(data_size):
		temp = []
		temp.extend(color[i])
		for j in range(len(histo[i])):
			temp.extend(histo[i][j])
		temp.append((pca_ima[i]))
		for j in range(len(square[i])):
			temp.extend(square[i][j])
		feature_data += [temp]
	print(feature_data)
	return feature_data

def get_label_data(file_name):
	f = open(file_name)
	content = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip()[0] for x in content]
	print(content)

# data = [color, histo, pca_ima, square]
def try_svm(color, histo, pca_ima, square):
	data_train = get_train_data(color, histo, pca_ima, square)
	data_label = get_label_data('/home/chaofeng/Documents/practicum/label.txt')

	data_train = np.asarray(data_train)
	data_label = np.asarray(data_label)
	c, r = data_label.shape
	data_label = data_label.reshape(c,)

	svm = SVC()
	score = cross_validation.cross_val_score(svm, data_train, data_label, cv = 10, scoring='accuracy')
	print(score)


# get_label_data('/home/chaofeng/Documents/practicum/label.txt')




########### Predict





# # importing necessary libraries

 
# # loading the iris dataset
# iris = datasets.load_iris()
 
# # X -> features, y -> label
# X = iris.data
# y = iris.target
 
# # dividing X, y into train and test data
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
 
# # training a linear SVM classifier
# from sklearn.svm import SVC
# svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
# svm_predictions = svm_model_linear.predict(X_test)
 
# # model accuracy for X_test  
# accuracy = svm_model_linear.score(X_test, y_test)
 
# # creating a confusion matrix
# cm = confusion_matrix(y_test, svm_predictions)