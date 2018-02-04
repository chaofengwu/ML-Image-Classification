from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import linear_model

exclude_wrong_file = 1
# Input data:
# 	color 5
# 	histo 3
# 	mean 3
# 	image_pca 10

########### Train

##### Get input data
def get_train_data(dimension, color, histo, pca_ima, square):
	data_size = len(color)
	feature_data = []
	data_choosed = []
	# print(color[54])
	for i in range(data_size):
		# print(i)
		# print(len(pca_ima))
		if exclude_wrong_file:
			if dimension[i] == [-1,-1]:
				continue
		temp = []
		# temp.extend(color[i])
		# for j in range(len(histo[i])):
		# 	# temp.append(histo[i][j][0])
		# 	temp.extend(histo[i][j])
		# for j in range(len(square[i])):
		# 	temp.extend(square[i][j])
		# temp.extend((pca_ima[i]))

		
		# temp.extend(color[i])
		# temp.extend([j/(color[i][len(color[i])-1]) for j in color[i]])
		# for j in range(len(histo[i])):
		# 	temp.append(histo[i][j][0])
		# 	# temp.extend(histo[i][j])
		# for j in range(len(square[i])):
		# 	temp.extend(square[i][j])
		temp.extend((pca_ima[i]))



		feature_data += [temp]
		data_choosed.append(i)
	# print(feature_data[0])
	# print(feature_data[32])
	# print(feature_data[53])
	# print(feature_data[80])
	return data_choosed, feature_data

def get_label_data(file_name, data_choosed):
	f = open(file_name)
	content = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip()[0] for x in content]
	res = []
	for i in range(len(data_choosed)):
		res.append(int(content[data_choosed[i]]))
	return res
	# print(content)

# data = [color, histo, pca_ima, square]
def try_svm(dimension, color, histo, pca_ima, square, label_file_name):
	[data_choosed, data_train] = get_train_data(dimension,color, histo, pca_ima, square)
	data_label = get_label_data(label_file_name, data_choosed)

	# print(data_train)
	# print(type(data_label))
	# print(len(data_train))
	# print(len(data_label))
	data_train = np.asarray(data_train)
	data_label = np.asarray(data_label)
	# print(data_label.shape)
	# c, r = data_label.shape
	# data_label = data_label.reshape(c,)

	logreg = linear_model.LogisticRegression(C=1e5)# .fit(data_train, data_label)
	score = cross_validation.cross_val_score(logreg, data_train, data_label, cv = 10, scoring='accuracy')
	print(sum(score)/len(score))
	# svm = SVC(kernel='rbf', C=1)
	# # cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
	# score = cross_val_score(svm, data_train, data_label, cv = 5)
	# score = cross_validation.cross_val_score(svm, data_train, data_label, cv = 10, scoring='accuracy')
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