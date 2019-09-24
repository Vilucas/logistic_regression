#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  logreg_predict.py                                                           #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Wednesday December 1969 4:00:00 pm                                 #
#  Modified: Tuesday Sep 2019 2:30:51 pm                                       #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import argparse
import pickle
import sys
import pandas as pd
import numpy as np
import src.MinMaxScaler as MinMaxScaler
import src.MultiLogReg as MultiLogReg

def parse():
	parser = argparse.ArgumentParser(description='Softmax Prediction for Multidimensional Data')
	parser.add_argument('model')
	parser.add_argument('data')
	return parser.parse_args()

if __name__ == '__main__':
	args = parse()

	# load model
	try:
		with open(args.model, 'rb') as model_file:
			model = pickle.load(model_file)
	except Exception as e:
		print(e)
		sys.exit(0)
	if not model:
		print('invalid model')

	# load test data
	try:
		data = pd.read_csv(args.data)
	except Exception as e:
		print(e)
		exit(1)

	# Seperate and prepare x values
	x = data.select_dtypes(include=[np.number])[['E', 'A','L', 'D', 'F']]
	x_num = x.to_numpy()
	scaler = MinMaxScaler.MinMaxScaler()
	scaler.load(model[1])
	x_transform = scaler.transform(x_num)

	# Instantiate the softmax classifier
	logreg = MultiLogReg.MultiLogReg()
	logreg.load(model[0])

	pred = pd.DataFrame(logreg.predict(x_transform), columns=['Category'])

	key = {0:'Group 0', 1:'Group 1', 2:'Group 2', 3:'Group 3'}
	pred.replace(key, inplace=True)
	pred.to_csv('categories.csv', index=True, index_label='Index')
