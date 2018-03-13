import model
import get_file_list
import data
import sys
import argparse
import json
import time
import numpy as np

parser = argparse.ArgumentParser(description='image classification: main function')
parser.add_argument('--mode', type=str, default='test',
                    help='mode to run the program: test, train, predict')
parser.add_argument('--folder_path', type=str, default='../copy_images',
                    help='the path of folder to work on')
parser.add_argument('--folder_mode', type=int, default=1,
                    help='whether looking through sub-folder, 1 is yes')
parser.add_argument('--label', type=str, default='',
                    help='if test, input the label file')
parser.add_argument('--image_path', type=str, default='',
					help='predict a image')
parser.add_argument('--prediction_file', type=str, default='prediction.json',
					help='where to store the prediction')
parser.add_argument('--resize_to', type=int, default=300,
					help='size of intermediate image')
parser.add_argument('--pca_components', type=int, default=30,
					help='dimension of reduced feature')
argv = parser.parse_args()


mode_flag = argv.mode
folder_path = argv.folder_path
flag = argv.folder_mode
label_file_path = argv.label
image_path = argv.image_path
prediction_file = argv.prediction_file
resize_size = argv.resize_to
pca_components = argv.pca_components

if(image_path == ''):
	print('get all image files')
	start_time = time.time()
	[file_list, file_name] = get_file_list.get_system_metadata(folder_path, flag)
	end_time = time.time()
	print('time used to get image files: ' + str(end_time-start_time))

else:
	if(mode_flag == 'predict'):
		print('get the image')
		start_time = time.time()
		file_list = [image_path]
		end_time = time.time()
		print('time used to get image: ' + str(end_time-start_time))
	else:
		print('if input just one image, it should be in predict mode')
		

print('get all image data')
start_time = time.time()
X, valid_list = data.get_image(file_list, resize_size)
end_time = time.time()
print('time used to extract image data: ' + str(end_time-start_time))
# print(X)
get_data_time = []

for i in range(len(file_list)):
	start_time = time.time()
	X, valid_list = data.get_image(file_list[0:i], resize_size)
	end_time = time.time()
	get_data_time.append(end_time-start_time)
	print('time used to extract image data: ' + str(end_time-start_time))

f = open('data_time.txt', 'w')
for i in get_data_time:
	f.write(str(i) + '\n')
f.close()


if(mode_flag == 'test'):
	print('get all label data')
	y = data.get_label_data(label_file_path, valid_list)
	start_time = time.time()
	model.test(X, y, resize_size, pca_components)
	end_time = time.time()
	print('finish')
	print('time used for test: ' + str(end_time-start_time))
elif(mode_flag == 'train'):
	print('get all label data')
	y = data.get_label_data(label_file_path, valid_list)
	print('start training')
	start_time = time.time()
	model.train(X, y, resize_size, pca_components)
	end_time = time.time()
	print('finish train, model in clf and pca')
	print('time used for train: ' + str(end_time-start_time))
elif(mode_flag == 'test_predict'):
	print('get all label data')
	y = data.get_label_data(label_file_path, valid_list)
	print('start test predict')
	start_time = time.time()
	model.test_predict(X, y)
	end_time = time.time()
	print('finish test_predict')
	print('time used for test_predict: ' + str(end_time-start_time))
elif (mode_flag == 'predict'):
	time_list = []
	# print(X[0:2])
	# for i in range(len(X)):
	# 	print('start predict')
	# 	start_time = time.time()
	# 	if i == 0:
	# 		tttt = X[0].reshape(1,-1)
	# 		prediction = model.predict(tttt)
	# 	else:
	# 		prediction = model.predict(X[0:i])
	# 	end_time = time.time()
	# 	time_list.append(end_time-start_time)
	# 	print('finish prediction')
	# 	print('time used to predict: ' + str(end_time-start_time))
	# print(time_list)
	# f = open('time.txt', 'w')
	# for i in time_list:
	# 	f.write(str(i) + '\n')
	# f.close()
	# print('save to ' + prediction_file)
	# data = []
	# for i in range(0, len(file_list)):
	# 	data.append({file_list[i]: int(prediction[i])})
	# # print(data)
	# with open(prediction_file, 'w') as json_file:
	# 	json.dump(data, json_file)

else:
	print('unknown mode')
