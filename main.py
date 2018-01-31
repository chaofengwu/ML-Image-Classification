import system_metadata
import image_processing
import time


time_flag = 1
# [file_list, file_name, file_extension, file_size] = get_system_metadata("/home/ubuntu/try", 1)
if time_flag:
	start_time = time.time()
print(start_time)
[file_list, file_name, file_extension, file_size] = system_metadata.get_system_metadata("/home/chaofeng/Documents/practicum/copy_images", 1)

image_matadata = []
count = 1
for i in file_list:
	image_matadata += [image_processing.pil_get_image_metadata(i, 1, 60)]
	# image_processing.get_image_metadata(i, 2, 3)
	print(str(count) + i + "\n")
	count += 1

if time_flag:
	end_time = time.time()
	train_time = end_time - start_time
	print('total time: %d seconds' % train_time)

print(len(file_list))
print(len(image_matadata))
print(image_matadata[0])
print(image_matadata[82])
# file_n = "file_list.txt"
# file = open(file_n, "w")
# for i in file_list:
# 	file.write(i + "\n")

# file.close()