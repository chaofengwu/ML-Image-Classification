import model
import get_file_list
import data

mode_flag = sys.argv[1]
folder_path = sys.argv[2]
try:
	flag = int(sys.argv[3])
except:
	flag = 1
label_file_path = sys.argv[4]


[file_list, file_name] = get_file_list.get_system_metadata(folder_path, flag)


X, valid_list = data.get_image(file_list)
y = data.get_label_data(label_file_path, valid_list)

if(mode_flag == 'test'):
	model.test(X, y)
else if(mode_flag == 'train'):
	model.train(X, y)
else if (mode_flag == 'predict'):
	model.predict(X)