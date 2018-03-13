from PIL import Image
import os
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json

plot_flag = 0
image_size = 300,300

def add(x,y): 
	return x+y


def getRed(redVal):
    return '#%02x%02x%02x' % (redVal, 0, 0)

def getGreen(greenVal):
    return '#%02x%02x%02x' % (0, greenVal, 0)

def getBlue(blueVal):
    return '#%02x%02x%02x' % (0, 0, blueVal)


def plot_histogram(image): # used for plot histogram
	histogram = image.histogram()
	# Take only the Red counts
	l1 = histogram[0:256]
	# Take only the Blue counts
	l2 = histogram[256:512]
	# Take only the Green counts
	l3 = histogram[512:768]
	plt.figure(0)
	# R histogram
	for i in range(0, 256):
	    plt.bar(i, l1[i], color = getRed(i), edgecolor=getRed(i), alpha=0.3)

	# G histogram
	plt.figure(1)
	for i in range(0, 256):
	    plt.bar(i, l2[i], color = getGreen(i), edgecolor=getGreen(i),alpha=0.3)

	# B histogram
	plt.figure(2)
	for i in range(0, 256):
	    plt.bar(i, l3[i], color = getBlue(i), edgecolor=getBlue(i),alpha=0.3)

	plt.show()


def get_color_histo(image): # get histogram
	image = image.convert('RGB')
	# print(image)

	# image = image.convert('L') # convert to grayscale
	# pix = image.load()
	# print(image.size)
	# image.show()
	# im = np.array(image)
	# print(im.shape)
	# print(im[0][0])
	if plot_flag:
		plot_histogram(image)
	return image.histogram()
	


# get_color_histo("maps.jpeg")
# get_color_histo("plots.gif")
# get_color_histo("maps.tiff")

def get_dimension(image):
	# print(image.size)
	return image.size

# get_dimension("maps.jpeg")

def get_mode(image): # RGB, P, etc
	return image.mode


def get_color(image): # Returns a list of colors used in this image.
	im = image.convert("L")
	return im.getcolors()


def get_extrema(image): # Gets the the minimum and maximum pixel values for each band in the image.
	return image.getextrema()


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
		# print('in n_square_mean: %d' % len(temp))
		# print('in n_square_mean: %d' % len(temp[0]))
		res.append(temp)
		# print('in n_square_mean: %d' % len(res[0]))
	# print('res: %d' % len(res))
	# print('res: %d' % len(res[0]))
	# print('res: %d' % len(res[0][0]))
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


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def pca_value(input_data, n_components):
	pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
	# pca = PCA(n_components='mle')

	# pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(input_data)
	a = pca.fit_transform(input_data)
	# eigenfaces = pca.components_.reshape((n_components, 300,300))
	# a = pca.transform(input_data)
	# print(a.explained_variance_ratio_)
	# print(a.singular_values_)
	return a#.singular_values_
	# return a


def pca_image(image, n_components):
	i = image.resize(image_size, Image.ANTIALIAS)
	# i.show()
	imag = image.convert("L")
	# ima = np.array(imag)
	# ima = np.array(list(imag.getdata()))
	ima = list(imag.getdata())
	return ima
	# return pca_value(ima, n_components).tolist()


def scale_to_width(dimensions, width):
    height = (width * dimensions[1]) / dimensions[0]

    return (width, height)


def pil_get_image_metadata(file_name, n_block, n_mean): #totally get n_block, each one is n_mean^2 square mean
	n_for_1 = n_mean*n_mean
	total_block = n_block*n_block
	try:
		image = Image.open(file_name)
	except:
		return [(-1,-1), "RGB", [],[],[]]

	dimension = get_dimension(image)
	mode = get_mode(image)
	# color = get_color(image)
	# extrema = get_extrema(image)
	histo = get_color_histo(image)
	# pca_ima = pca_image(image, 10)
	# print(pca_ima)
	# coeffs = find_coeffs([(0,0), (image_size[0], 0), (image_size[0], image_size[1]), (0, image_size[1])],[(0,0), (image_size[0], 0), (image_size[0], image_size[1]), (0, image_size[1])])

	try:
		im = image.resize(image_size, Image.ANTIALIAS)
		# im = image.transform(image_size, Image.PERSPECTIVE, coeffs, resample = Image.BICUBIC)
		# im.show()
		dime = get_dimension(im)
		square = get_n_block(im, dime, n_block, n_mean)
	except:
		print("error when transfrom image size in PIL: " + file_name)
		square = []
	# return [dimension, mode, color, histo, square, extrema]
	return [dimension, mode, histo, square]


# [dimension, mode, color, extrema, histo, square] = get_image_metadata("maps.jpeg", 2, 3)
# print(dimension)
# print(color)
# print(len(square[0][0]))