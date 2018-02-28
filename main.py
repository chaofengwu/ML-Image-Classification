import system_metadata
# import image_processing
import image_metadata
import svm
import time


time_flag = 1
# [file_list, file_name, file_extension, file_size] = get_system_metadata("/home/ubuntu/try", 1)
if time_flag:
	start_time = time.time()
print(start_time)

###### system file metadata

[file_list, file_name, file_extension, file_size] = system_metadata.get_system_metadata("/home/chaofeng/Documents/practicum/copy_images", 1)

###### image content prediction

[dimension, mode, color, histo, square] = image_metadata.image_metadata(file_list)

# print(color[0])
# print(histo[0])
# print(len(histo))
# print(len(histo[0]))
# print(len(histo[0][0]))
# print(len(histo[0][0][0]))
# # # print(len(pca_ima[0][1]))
# # print(len(histo[0]))
# print(len(square[0]))
svm.try_svm('/home/chaofeng/Documents/practicum/file_list.txt', '/home/chaofeng/Documents/practicum/data.json', '/home/chaofeng/Documents/practicum/label_no_wrong_file.txt')
# svm.try_logi_reg(file_list, dimension, color, histo, square, '/home/chaofeng/Documents/practicum/test_label.txt')

# input_data = [dimension, color, histo, pca_ima, square]
# raw_image_matadata = []
# count = 1
# for i in file_list:
# 	raw_image_matadata += [image_processing.pil_get_image_metadata(i, 1, 100)]
# 	# image_processing.get_image_metadata(i, 2, 3)
# 	print(str(count) + i + "\n")
# 	count += 1

if time_flag:
	end_time = time.time()
	train_time = end_time - start_time
	print('total time: %d seconds' % train_time)

# print(len(file_list))
# print(len(dimension))
# print(dimension)
# print(color)
# print(histo)
# print(pca_ima)
# print(square)

# file_n = "file_list.txt"
# file = open(file_n, "w")
# for i in file_list:
# 	file.write(i + "\n")

# file.close()