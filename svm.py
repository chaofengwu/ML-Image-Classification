from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
# from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import linear_model
from PIL import Image
import modify_image_data

exclude_wrong_file = 1
# Input data:
# 	color 5
# 	histo 3
# 	mean 3
# 	image_pca 10

########### Train

##### Get input data
iamge_size = (100,100)
def read_file_list(file_name):
	with open(file_name) as f:
		content = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
		content = [x.strip() for x in content]
	return content


def get_image_pca(file_name):
	X = []
	file_list = read_file_list(file_name)
	valid_list = []
	idx = 0
	for i in file_list:
		try:
			image = Image.open(i)
			image = image.resize(iamge_size, Image.ANTIALIAS)
			image = image.convert('L')
			img_array = list(image.getdata())
			X.append(img_array)
			# valid_list.append(idx)
			
		except:
			# pass
			X.append([])
		# idx += 1

	return X# , valid_list


def get_train_data(data_file, pca_ima):
	[dimension, color, histo, square] = modify_image_data.modify_image_metadata()
	data_size = len(color)
	feature_data = []
	data_choosed = []
	# print(color[54])
	# print(color[0])
	# print(histo[0])
	# print(pca_ima[0])
	# print(square[0])
	for i in range(data_size):
		# print(i)
		# print(len(pca_ima))
		if exclude_wrong_file:
			if dimension[i] == [-1,-1]:
				continue
		temp = []
		# temp.extend(color[i])
		# for j in range(len(histo[i])):
		# 	temp.extend(histo[i][0][j])
		# for j in range(len(square[i])):
		# 	temp.extend(square[i][j])
		temp.extend((pca_ima[i]))


		feature_data += [temp]
		data_choosed.append(i)
	# print(feature_data[0])
	# print(feature_data[32])
	# print(feature_data[53])
	# print(feature_data[80])
	# print(feature_data[1][0])
	return data_choosed, feature_data

def get_label_data(file_name, data_choosed):
	f = open(file_name)
	content = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip()[0] for x in content]
	res = []
	for i in range(len(content)):
		if content[i] != '0':
			res.append(int(content[i]))
	# for i in range(len(data_choosed)):
	# 	res.append(int(content[data_choosed[i]]))
	return res
	# print(content)

# data = [color, histo, pca_ima, square]
def try_svm(file_list, data_file, label_file_name):
	pca_ima = get_image_pca(file_list)
	# print(pca_ima)
	[data_choosed, data_train] = get_train_data(data_file, pca_ima)
	data_label = get_label_data(label_file_name, data_choosed)

	# print(data_train[0])
	# print(data_train)
	# print(type(data_label))
	# print(len(data_train))
	# print(len(data_label))
	data_train = np.asarray(data_train)
	data_label = np.asarray(data_label)

	X_train, X_test, y_train, y_test = train_test_split(data_train, data_label, test_size = 0.5, random_state=42)
	n_components = 50
	pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

	eigenfaces = pca.components_.reshape((n_components, 100, 100))

	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	# print(X_test_pca[0])
	param_grid = {'C': [1e3,5e3,1,4,5e4,1e5], 'gamma': [1e-4, 5e-4,1e-3,5e-3,1e-2,1e-1], }
	clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid).fit(X_train_pca, y_train)
	clf = clf.fit(X_train_pca, y_train)

	print(clf.best_estimator_)


	y_pred = clf.predict(X_test_pca)
	count = 0
	for i in range(len(y_test)):
		if y_pred[i] == y_test[i]:
			count += 1
	
	print(y_pred)
	print(y_test)
	print(count/len(y_test))


	# # print(data_label.shape)
	# # c, r = data_label.shape
	# # data_label = data_label.reshape(c,)

	# # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	# # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
	# # 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	# param_grid = {'C': [1e3,5e3,1,4,5e4,1e5], 'gamma': [1e-4, 5e-4,1e-3,5e-3,1e-2,1e-1], }
	# svm = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid).fit(data_train, data_label)
 #    # svm = SVC(kernel='rbf')
	# # svm = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	# # score = cross_val_score(svm, data_train, data_label, cv = 10)
	# score = cross_validation.cross_val_score(svm, data_train, data_label, cv = 10, scoring='accuracy')
	# print(score)
	# print(sum(score)/len(score))

def try_logi_reg(dimension, color, histo, pca_ima, square, label_file_name):
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

	logreg = linear_model.LogisticRegression(C=1e6)
	score = cross_val_score(logreg, data_train, data_label, cv = 10, scoring='accuracy')
	print(score)
	print(sum(score)/len(score))


# get_label_data('/home/chaofeng/Documents/practicum/label.txt')




########### Predict
