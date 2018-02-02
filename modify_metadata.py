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

def add(x,y): 
	return x+y

###### for dimension, nothing to do ###### 2

###### for mode, nothing to do ###### 1

###### for color, divided into group, 255-251, 250-169, 168-87, 86-5, 4-0 ###### 5

###### for histo, PCA to 6 ###### 6

###### for mean, may get avarage pixel ###### 3

def modify_color(co):
	res = [0,0,0,0,0]
	temp = 0
	for idx in range(len(co)):
		a = 1
		if (co[idx][1]) < 5:
			res[0] += co[idx][0]
		elif co[idx][1] < 87:
			res[1] += co[idx][0]
		elif co[idx][1] < 169:
			res[2] += co[idx][0]
		elif co[idx][1] < 251:
			res[3] += co[idx][0]
		elif co[idx][1] < 256:
			res[4] += co[idx][0]
		else:
			print('modify color: color index out of range')
	return res


def modify_histogram(hi):
	res = []
	temp = [hi[0:255],hi[255:510],hi[510:765]]
	# print(temp)
	for i in range(3):
		res += [pca_value([temp[i],range(255)],2)]
	return res


def modify_square(sq):
	h = len(sq)
	w = len(sq[0])
	total = h*w
	res = [0,0,0]
	for i in range(h):
		for j in range(w):
			res = map(add, (sq[i][j]), res)
	return [i/total for i in res]



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
		print(str(count) + i + "\n")
		[di, mo, co, hi, pca_ima, sq] = image_processing.pil_get_image_metadata(i, 1, 100)
		if di != (-1,-1):
			co = modify_color(co)
			hi = modify_histogram(hi)
			sq = modify_square(sq)
		dimension += [di]
		mode += [mo]
		color += [co]
		histo += [hi]
		square += [sq]
		# image_processing.get_image_metadata(i, 2, 3)
		
		count += 1
		# if count == 3:
		# 	break
	# print(histo)
	
	return [dimension, mode, color, histo, pca_ima, square]

# modify_image_metadata(['/home/chaofeng/Documents/practicum/copy_images/images/._RR2016.jpg'])