from PIL import Image

def get_label_data(file_name):
	f = open(file_name)
	content = f.readlines()
	
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = list([list([x[0], x[2:len(x)-1]]) for x in content])
	# print(content)
	f.close()
	return content
	

def get_wrong_data(file_name, output_file):
	content = get_label_data(file_name)
	f = open(output_file, 'w')
	print(content[0])
	count = 1
	for i in content:
		if i[0] == '6':
			try:
				print(str(count) + ':' + i[1])
				image = Image.open(i[1])
				image.show()
				t = input('need change: 1 need select: 2 not need: 3: ')
				if t == '1':
					f.write('3\t' + i[1] + '\n')
				if t == '2':
					choice = input('1 line_plots, 2 maps, 3 map&depth_chart, 4 map&colorplot, 5 map&histogram, 6 for figures: ')
					f.write(choice + '\t')
					f.write(i[1] + '\n')
				else:
					pass
			except:
				pass
			count += 1
			# if count == 10:
			# 	break
	f.close()


def change_data(file_name, correct_file, new_file):
	origin = get_label_data(file_name)
	corrected = get_label_data(correct_file)
	new = open(new_file,'w')
	count = 1
	i = 0
	j = 0
	while i < len(corrected) and j < len(origin):
		print(corrected[i][1])
		print(origin[j][1])
		if corrected[i][1] == origin[j][1]:
			print(corrected[i][1])
			new.write(corrected[i][0] + '\t' + origin[j][1] + '\n')
			i += 1
			j += 1
		else:
			new.write(origin[j][0] + '\t' + origin[j][1] + '\n')
			j += 1
	if i != len(corrected):
		print('i value error: %d' % i)
	new.close()


# get_wrong_data('/home/chaofeng/Documents/practicum/label.txt', 'changed_label.txt')
# change_data('/home/chaofeng/Documents/practicum/label.txt', 'changed_label.txt', 'new_label.txt')

def change_label(file_name, output_file):
	content = get_label_data(file_name)
	f = open(output_file, 'w')
	for i in content:
		if i[0] == '5':
			f.write('4\t' + i[1] + '\n')
		else:
			f.write(i[0] + '\t' + i[1] + '\n')
	f.close()

change_label('test_label.txt', 'test_label1.txt')