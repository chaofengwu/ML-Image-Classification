import numpy as np
import image_processing
from sklearn.decomposition import PCA


def pca_value(input_data, n_components):
	pca = PCA(n_components=n_components)
	# pca = PCA()
	a = pca.fit(input_data)
	# print(a.explained_variance_ratio_)
	# print(a.singular_values_)
	return a.singular_values_


###### for dimension, nothing to do ###### 2

###### for mode, nothing to do ###### 1

###### for color, divided into group, 255-251, 250-169, 168-87, 86-5, 4-0 ###### 5

###### for histo, PCA to 6 ###### 6

###### for mean, may get avarage pixel ###### 3

def modify_color(color):
	res = []
	temp = 0
	for i in range(5):
		temp += color[i][0]
	res += [temp]
	temp = 0
	for i in range(5,87):
		temp += color[i][0]
	res += [temp]
	temp = 0
	for i in range(87,169):
		temp += color[i][0]
	res += [temp]
	temp = 0
	for i in range(169,251):
		temp += color[i][0]
	res += [temp]
	temp = 0
	for i in range(251,256):
		temp += color[i][0]
	res += [temp]
	return res


def modify_histogram(histo):
	res = []
	for i in range(len(histo)):
		temp = [[histo[i][0:255]],[histo[i][255:510]],[histo[i][510:765]]]
		res += [pca_value(temp,6)]
	return res


def modify_square(square):
	h, w = square.size
	res = [0,0,0]
	for i in range(h):
		for j in range(w):
			t = map(add, (square[i][j]), res)
	return res



def modify_image_metadata(file_list):
	###### First get raw image data and then modify them
	raw_image_matadata = []
	dimension = []
	mode = []
	color = []
	histo = []
	square = []
	count = 1
	for i in file_list:
		[di, mo, co, hi, pca_ima, sq] = image_processing.pil_get_image_metadata(i, 1, 100)
		dimension += di
		mode += mo
		color += co
		histo += hi
		square += sq
		# image_processing.get_image_metadata(i, 2, 3)
		print(str(count) + i + "\n")
		count += 1
		if count == 6:
			break
	print(histo[0])
	color = modify_color(color)
	histo = modify_histogram(histo)
	square = modify_square(square)
	return [dimension, mode, color, histo, pca_ima, square]
