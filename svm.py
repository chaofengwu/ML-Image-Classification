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
import time
from sklearn.metrics import precision_recall_fscore_support
exclude_wrong_file = 0
# Input data:
# 	color 5
# 	histo 3
# 	mean 3
# 	image_pca 10

########### Train

##### Get input data
image_size = (300,300)
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
	count = 1
	for i in file_list:
		# print(str(count) + i + "\n")
		try:
			image = Image.open(i)
			image = image.resize(image_size, Image.ANTIALIAS)
			image = image.convert('L')
			img_array = list(image.getdata())
			X.append(img_array)
			
		except:
			X.append([])
		count += 1
	return X


def get_histo(file_name):
	X = []
	file_list = read_file_list(file_name)
	valid_list = []
	idx = 0
	count = 1
	for i in file_list:
		# print(str(count) + i + "\n")
		try:
			image = Image.open(i)
		except:
			print('in get_histo, cannot open file')
		image = image.convert('RGB')
		# if plot_flag:
		# 	plot_histogram(image)
		X += [image.histogram()]
		count += 1
	return X

def cut_edge(upper_b, lower_b, right_b, left_b, n): # Gets suitable size of block
	diff = (upper_b - lower_b) % n
	if diff != 0:
		upper_b -= diff

	diff = (right_b - left_b) % n
	if diff != 0:
		right_b -= diff

	col_length = int((right_b - left_b) / n)
	row_length = int((upper_b - lower_b) / n)
	return [upper_b, right_b, col_length, row_length]


def n_square_mean(image, upper_b, lower_b, right_b, left_b, n, mode):
	if mode == "RGB":
		image = image.convert("RGB")
	else:
		image = image.convert("L")
	pix = image.load()
	res = []
	############cut the image to same blocks#################
	[upper_b, right_b, col_length, row_length] = cut_edge(upper_b,lower_b,right_b,left_b,n)
	sq = n*n
	##########Add them up###################
	for i in range(row_length):
		temp = []
		for j in range(col_length):
			if mode == "RGB":
				t = [0,0,0]
			else:
				t = [0]
			for k in range(n):
				for l in range(n):
					try:
						t = map(add, (pix[lower_b + i + k*row_length,left_b + j + l*col_length]), t)
					except:
						# t = [0,0,0]
						if mode == "L":
							t[0] = pix[lower_b + i + k*row_length,left_b + j + l*col_length] + t[0]
						else:
							print([i + k*row_length,j + l*col_length])
							print("index wrong in n_square_mean")
			temp.append([x/sq for x in t])
		# print('in n_square_mean: %f' % len(temp))
		# print('in n_square_mean: %f' % len(temp[0]))
		res.append(temp)
		# print('in n_square_mean: %f' % len(res[0]))
	# print('res: %f' % len(res))
	# print('res: %f' % len(res[0]))
	# print('res: %f' % len(res[0][0]))
	return res


def get_n_block(image, dimension, n_block, n_mean):
	square = []
	# [upper_b, right_b, col_length, row_length] = cut_edge(dimension[0],0,dimension[1], 0, n_block)
	row_length = int(dimension[0] / n_block)
	col_length = int(dimension[1] / n_block)
	for i in range(n_block):
		for j in range(n_block):
			square += [n_square_mean(image, dimension[0] - i*row_length, dimension[0]-(i+1)*row_length, dimension[1]-j*col_length, dimension[1] - (j+1)*col_length, n_mean, "L")]
	return square


def get_square(file_name, n_block, n_mean):
	X = []
	file_list = read_file_list(file_name)
	valid_list = []
	idx = 0
	count = 1
	for i in file_list:
		# print(str(count) + i + "\n")
		try:
			image = Image.open(i)
			im = image.resize(image_size, Image.ANTIALIAS)
		except:
			print('in get_square, cannot open file')
		dime = im.size
		square = get_n_block(im, dime, n_block, n_mean)

		X += [square]
		count += 1
	return X


def get_train_data(file_list, flag):
	if flag & 1:
		data = get_histo(file_list)
	elif flag & 2:
		data = get_square(file_list, 3, 3)
	elif flag & 4:
		data = get_image_pca(file_list)
	# [dimension, histo, square] = modify_image_data.modify_image_metadata()
	data_size = len(data)
	feature_data = []
	data_choosed = []
	# print(color[54])
	# print(color[0])
	# print(histo[0])
	# print(extrema[0])
	# print(pca_ima[0])
	# print(square[54])
	for i in range(data_size):
		# print(i)
		# print(len(pca_ima))
		if exclude_wrong_file:
			if dimension[i] == [-1,-1]:
				continue
		temp = []
		if flag == 1:
			temp.extend(data[i])
		elif flag == 2:
			for j in range(len(data[i])):
				for k in range(len(data[i][j])):
					for l in range(len(data[i][j][k])):
						temp.extend(data[i][j][k][l])
		elif flag == 4:
			temp.extend(data[i])
		# for j in range(len(square[i])):
		# 	for k in range(len(square[i][j])):
		# 		for l in range(len(square[i][j][k])):
		# 			temp.extend(square[i][j][k][l])
		# temp.extend((pca_ima[i]))

		# temp.extend(color[i])
		# for j in range(len(histo[i])):
		# 	temp.extend(histo[i][0][j])
		


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
def try_svm(file_list, label_file_name, flag):
	
	# print(pca_ima)
	start_time = time.time()
	[data_choosed, data_train] = get_train_data(file_list, flag)
	end_time = time.time()
	print('time to get image data: %f seconds' % (end_time-start_time))
	start_time = time.time()
	data_label = get_label_data(label_file_name, data_choosed)

	print(len(data_train[0]))
	print(type(data_train[0]))
	# print(data_train[0])
	# print(type(data_label))
	# print(len(data_train))
	# print(len(data_label))
	data_train = np.asarray(data_train)
	data_label = np.asarray(data_label)

	X_train, X_test, y_train, y_test = train_test_split(data_train, data_label, test_size = 0.5, random_state=42)
	n_components = 30
	pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

	# if flag == 1:
	# 	eigenfaces = pca.components_.reshape((n_components, 256, 3)) # for histo
	# elif flag == 2:
	# 	eigenfaces = pca.components_.reshape((n_components, 99, 99)) # for square
	# elif flag == 4:
	# 	eigenfaces = pca.components_.reshape((n_components, 300, 300)) # for pca_image	
	
	
	

	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	# print(X_test_pca[0])
	# param_grid = {'C': [1e3,5e3,1,4,5e4,1e5], 'gamma': [1e-4, 5e-4,1e-3,5e-3,1e-2,1e-1], }
	param_grid = {'C': [1e3,5e3,1,4,5e4,1e5,1e6], 'gamma': [1e-4, 5e-4,1e-3,5e-3,1e-2,1e-1], }
	clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid).fit(X_train_pca, y_train)
	clf = clf.fit(X_train_pca, y_train)
	end_time = time.time()
	print('time to train: %f seconds' % (end_time-start_time))
	start_time = time.time()
	# print(clf.best_estimator_)


	y_pred = clf.predict(X_test_pca)
	end_time = time.time()
	
	count = 0
	for i in range(len(y_test)):
		if y_pred[i] == y_test[i]:
			count += 1
	
	print(y_pred)
	print(y_test)
	print(count/len(y_test))
	print('time to train and predict: %f seconds' % (end_time-start_time))
	precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred)
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fbeta_score))
	print('support: {}'.format(support))
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

def try_logi_reg(file_list, label_file_name, flag):
	start_time = time.time()
	[data_choosed, data_train] = get_train_data(file_list, flag)
	end_time = time.time()
	print('time to get image data: %f seconds' % (end_time-start_time))
	
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



	print(len(data_train[0]))
	print(type(data_train[0]))
	# print(data_train[0])
	# print(type(data_label))
	# print(len(data_train))
	# print(len(data_label))
	data_train = np.asarray(data_train)
	data_label = np.asarray(data_label)

	X_train, X_test, y_train, y_test = train_test_split(data_train, data_label, test_size = 0.5, random_state=42)
	n_components = 30
	start_time = time.time()
	pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

	if flag == 1:
		eigenfaces = pca.components_.reshape((n_components, 256, 3)) # for histo
	elif flag == 2:
		eigenfaces = pca.components_.reshape((n_components, 99, 99)) # for square
	elif flag == 4:
		eigenfaces = pca.components_.reshape((n_components, 300, 300)) # for pca_image	
	
	
	

	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)

	param_grid = {'C': [1,1e1,1e2,1e3,5e3,1,4,5e4,1e5,1e6,1e10], }
	clf = GridSearchCV(linear_model.LogisticRegression(C=1e6), param_grid).fit(X_train_pca, y_train)
	clf = clf.fit(X_train_pca, y_train)
	# clf = linear_model.LogisticRegression(C=1e2)
	# clf = clf.fit(X_train_pca, y_train)

	# print(clf.best_estimator_)
	end_time = time.time()
	print('time to train: %f seconds' % (end_time-start_time))
	start_time = time.time()
	y_pred = clf.predict(X_test_pca)

	end_time = time.time()
	
	count = 0
	for i in range(len(y_test)):
		if y_pred[i] == y_test[i]:
			count += 1
	
	print(y_pred)
	print(y_test)
	print(count/len(y_test))
	print('time to train and predict: %f seconds' % (end_time-start_time))
	precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred)
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fbeta_score))
	print('support: {}'.format(support))
	# logreg = linear_model.LogisticRegression(C=1e6)
	# score = cross_val_score(logreg, data_train, data_label, cv = 10, scoring='accuracy')
	# print(score)
	# print(sum(score)/len(score))


# get_label_data('/home/chaofeng/Documents/practicum/label.txt')




########### Predict
