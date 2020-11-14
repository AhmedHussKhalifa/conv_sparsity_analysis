# PLots the total number of non zero elements for each conv nodes across images

import os
import operator
import matplotlib.pyplot as plt

MAIN_PATH       = '../gen/IV3_mean_dataset/'
conv_stat_files = os.listdir(MAIN_PATH)

total_nodes_count = len(list(conv_stat_files)) 
for ic in range(total_nodes_count):
# for ic in range(0, 2):
	full_file = os.path.join(MAIN_PATH, 'IV3_Conv_' + str(ic) + '.stat')

	mean_list = []
	std_list = []
	nze_list = []
	# Read files
	with open(full_file, 'r') as f: 
		lines =  f.readlines()


		Ih, Iw, Kh, Kw, Ic = int(lines[0].split('\t')[0]), int(lines[0].split('\t')[1]), \
			int(lines[0].split('\t')[2]), int(lines[0].split('\t')[3]), int(lines[0].split('\t')[4])
		
		for line in lines[1:-1]:
			mean_list.append(float(line.split('\t')[0]))
			std_list.append(float(line.split('\t')[1]))
			nze_list.append(int(line.split('\t')[2]))
		

	# Plotting Code
	images_indices = [i for i in range(len(lines[1:-1]))]


	# Plotting Code:
	plt.figure()
	plt.plot(nze_list, '-*')
	plt.xlabel('Image Index', color='k', size=16)
	plt.ylabel('Total NZE Count', color='k', size=16)
	tit = 'Conv_%d_%d_%d_%d_%d' % (ic, Ih, Iw, Kh, Kw)
	plt.title(tit)
	plt.xticks(size = 16, color='k')
	plt.yticks(size = 16, color='k')

	# plt.show()
	plt.savefig(MAIN_PATH + 'plots/' + tit + '.png', dpi=600, bbox_inches='tight')

	if not ic % 10:
		print('Done %d/%d ' % (ic, total_nodes_count))
