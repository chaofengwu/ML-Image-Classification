import system_metadata
import image_processing
import modify_metadata
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

[dimension, mode, color, histo, pca_ima, square] = modify_metadata.modify_image_metadata(file_list)


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

print(len(file_list))
print(len(dimension))
print(dimension[0])
print(color[0])
print(histo[0])
print(pca_ima[0])
print(square[0])
# file_n = "file_list.txt"
# file = open(file_n, "w")
# for i in file_list:
# 	file.write(i + "\n")

# file.close()