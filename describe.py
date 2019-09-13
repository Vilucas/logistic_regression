#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  describe.py                                                                 #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Wednesday December 1969 4:00:00 pm                                 #
#  Modified: Thursday Sep 2019 8:14:00 pm                                      #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import sys
import math

def usage():
	print('usage: ./describe.py data_file')

def harvest(data, func, row_name):
	h_vals = []
	for col in data:
		values = [val[1] for val in data[col].dropna().iteritems()]
		if not values:
			h_vals.append('NaN')
			continue
		h_vals.append(func(values))
	return pd.DataFrame([h_vals], columns=data.columns, index=[row_name])

def f_mean(data):
	return (sum(data) / len(data))

def f_std(data):
	m = f_mean(data)
	d = list(map(lambda x: (x - m) ** 2, data))
	return np.sqrt(sum(d) / (len(d) - 1))

def f_qntl(data, q):
	i = (len(data)) * q
	if i != int(i):
		return sorted(data)[int(math.floor(i))]
	else:
		s = sorted(data)
		i = int(i) - 1
		return (s[i] + s[i + 1]) / 2

def describe(data, assets):
	numeric_data = data.select_dtypes(include=[np.number])
	describe = pd.DataFrame(columns=numeric_data.columns, dtype='float')
	for asset in assets:
		describe = describe.append(harvest(numeric_data, asset[0], asset[1]))
	return describe

if __name__ == '__main__':
	if (len(sys.argv)) != 2:
		usage()
		exit(0)
	try:
		data = pd.read_csv(sys.argv[1])
	except Exception as e:
		print(e)
		exit(1)
	assets = [(len, 'count'),
		(f_mean, 'mean'),
		(f_std, 'std'),
		(min, 'min'),
		(lambda x: f_qntl(x, 0.25), '25%'),
		(lambda x: f_qntl(x, 0.5), '50%'),
		(lambda x: f_qntl(x, 0.75), '75%'),
		(max, 'max')]
	print(data.describe())
	print(describe(data, assets))
	# print(data)
