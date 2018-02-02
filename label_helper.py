import os
from PIL import Image


def read_file_list(file_name):
	with open(file_name) as f:
		content = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
		content = [x.strip() for x in content]
	return content


def label_file(content, output_file):
	f = open(output_file, 'w')
	count = 0
	for i in content:
		print('this is the %d images: ' % count + i)
		count += 1
		try:
			image = Image.open(i)
			image.show()
			choice = input('1 line_plots, 2 maps, 3 map&depth_chart, 4 map&colorplot, 5 map&histogram, 6 for figures: ')
			f.write(choice + '\t')
			f.write(i + '\n')

		except:
			f.write('0\t') # 0 for cannot open
			f.write(i + '\n')
	f.close()

file_list = read_file_list('file_list.txt')
label_file(file_list, 'label.txt')