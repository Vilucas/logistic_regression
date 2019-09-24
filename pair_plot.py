#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  pair_plot.py                                                                #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Wednesday December 1969 4:00:00 pm                                 #
#  Modified: Tuesday Sep 2019 2:22:27 pm                                       #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

HOUSE = ['Group 1', 'Group 2', 'Group 3', 'Group 0']

def usage():
	print('usage: ./describe.py data_file')

def scatter(X, Y):
	plt.scatter(x=X, y=Y)

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
	sample['Category'] = data['Category']

	data_col_len = len(sample.columns) - 1
	fig = plt.figure(figsize=(11.5,8))
	for row in range(1, data_col_len):
		for col in range(1, data_col_len):
			fig.add_subplot(data_col_len, data_col_len, data_col_len * row + col)
			plt.xticks([])
			plt.yticks([])
			if row == col:
				col_name = sample.columns[col - 1]
				for h in HOUSE:
					segment = sample[sample['Category'] == h][col_name]
					segment = segment[~np.isnan(segment)]
					plt.hist(segment, alpha=0.5, bins=20)
					plt.title(col_name, fontsize='x-small')
			else:
				col_name = sample.columns[col - 1]
				row_name = sample.columns[row - 1]
				for h in HOUSE:
					segment = sample[sample['Category'] == h]
					X = segment[row_name]
					Y = segment[col_name]
					plt.scatter(x=X, y=Y, alpha=0.5, s=2)
	plt.subplots_adjust(top=1, bottom=0.05, hspace = 0.5)
	fig.legend(HOUSE)
	plt.show()