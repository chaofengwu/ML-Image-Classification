from PIL import Image
import os
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import json

plot_flag = 1

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
	ima = Image.open(image)
	imag = ima.convert('RGB')
	# print(image)

	# image = image.convert('L') # convert to grayscale
	# pix = image.load()
	# print(image.size)
	# image.show()
	# im = np.array(image)
	# print(im.shape)
	# print(im[0][0])
	if plot_flag:
		plot_histogram(imag)
	return imag.histogram()

get_color_histo('111.jpg')