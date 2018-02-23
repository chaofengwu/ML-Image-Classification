import model
import get_file_list
import data
import sys
import argparse

parser = argparse.ArgumentParser(description='image classification: main function')
parser.add_argument('--mode', type=str, default='test',
                    help='mode to run the program: test, train, predict')
parser.add_argument('--folder_path', type=str, default='/',
                    help='the path of folder to work on')
parser.add_argument('--folder_mode', type=int, default=1,
                    help='whether looking through sub-folder, 1 is yes')
parser.add_argument('--label', type=str, default='',
                    help='if test, input the label file')
argv = parser.parse_args()


mode_flag = argv.mode
folder_path = argv.folder_path
flag = argv.folder_mode
label_file_path = argv.label

print('get all image files')
[file_list, file_name] = get_file_list.get_system_metadata(folder_path, flag)

print('get all image data')
X, valid_list = data.get_image(file_list)
# print(X)


if(mode_flag == 'test'):
	print('get all label data')
	y = data.get_label_data(label_file_path, valid_list)
	model.test(X, y)
	print('finish')
elif(mode_flag == 'train'):
	print('get all label data')
	y = data.get_label_data(label_file_path, valid_list)
	model.train(X, y)
	print('finish train, model in clf and pca')
elif(mode_flag == 'test_predict'):
	print('get all label data')
	y = data.get_label_data(label_file_path, valid_list)
	model.test_predict(X, y)
	print('finish test_predict')
elif (mode_flag == 'predict'):
	model.predict(X)
	print('finish prediction')
else:
	print('unknown mode')
