#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  scatter_plot.py                                                             #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Wednesday December 1969 4:00:00 pm                                 #
#  Modified: Friday Sep 2019 4:48:39 pm                                        #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

HOUSE = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']

def usage():
	print('usage: ./describe.py data_file')

if __name__ == '__main__':
	if (len(sys.argv)) != 2:
		usage()
		exit(0)
	try:
		data = pd.read_csv(sys.argv[1])
	except Exception as e:
		print(e)
		exit(1)
	sample = data.select_dtypes(include=[np.number]).iloc[:,1:]
	sample['Hogwarts House'] = data['Hogwarts House']
	fig = plt.figure(figsize=(10,8))
	i = 0
	label = []
	for col in sample.columns:
		if col == 'Hogwarts House' or col == 'Astronomy':
			continue
		i += 1
		fig.add_subplot(3, 4, i)
		plt.xticks([])
		plt.yticks([])
		for h in HOUSE:
			segment = sample[sample['Hogwarts House'] == h]
			X = segment['Astronomy']
			Y = segment[col]
			l = plt.scatter(x=X, y=Y, alpha=0.5, s=2)
			label.append(l)
			plt.title(col)
	plt.subplots_adjust(top=0.9, bottom=0.1, right=0.85, hspace = 0.5)
	plt.suptitle('Astonomy vs. Others')
	fig.legend(HOUSE)
	plt.show()