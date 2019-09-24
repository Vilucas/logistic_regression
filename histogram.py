#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  histogram.py                                                                #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Wednesday December 1969 4:00:00 pm                                 #
#  Modified: Tuesday Sep 2019 2:18:23 pm                                       #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

HOUSE = ['Group 1', 'Group 2', 'Group 3', 'Group 0']

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
	sample['Category'] = data['Category']
	label = []
	fig = plt.figure(figsize=(14.5,8))
	for i in range(1, 14):
		fig.add_subplot(4, 4, i)
		plt.xticks([])
		col = sample.columns[i - 1]
		for h in HOUSE:
			segment = sample[sample['Category'] == h][col]
			segment = segment[~np.isnan(segment)]
			l = plt.hist(segment, alpha=0.5, bins=20)
			label.append(l)
			plt.title(col)
	fig.legend(HOUSE)
	plt.show()
