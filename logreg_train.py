#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  DSLR.py                                                                     #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Wednesday December 1969 4:00:00 pm                                 #
#  Modified: Monday Sep 2019 2:30:04 pm                                        #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import argparse
import pickle
import numpy as np
import pandas as pd
import src.MinMaxScaler as MinMaxScaler
import src.MultiLogReg as MultiLogReg
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'

def parse():
	parser = argparse.ArgumentParser(description='Softmax Model for Wizard Hatting')
	parser.add_argument('-t', '--test', action='store_true')
	parser.add_argument('-p', '--plot', action='store_true')
	parser.add_argument('data')
	return parser.parse_args()

def save_model(model, model_name='dslr_model.pkl'):
	with open(model_name, mode='wb') as model_file:
		pickle.dump(model, model_file)

if __name__ == '__main__':
	args = parse()
	try:
		data = pd.read_csv(args.data)
	except Exception as e:
		print(e)
		exit(1)
	data = data.dropna()

	# Seperate y and prepare y values
	y = data['Hogwarts House']
	y.loc[y == 'Hufflepuff'] = 0
	y.loc[y == 'Slytherin'] = 1
	y.loc[y == 'Gryffindor'] = 2
	y.loc[y == 'Ravenclaw'] = 3

	# Seperate and prepare x values
	x = data.select_dtypes(include=[np.number])[['Divination', 'Arithmancy','Charms', 'Defense Against the Dark Arts', 'Muggle Studies']]
	x_num = x.to_numpy()
	scaler = MinMaxScaler.MinMaxScaler()
	scaler.fit(x_num)
	x_transform = scaler.transform(x_num)

	# Instantiate the softmax classifier
	logreg = MultiLogReg.MultiLogReg(plot=args.plot)

	# Train the model, possibly splitting to test
	if args.test:
		x_train, x_test, y_train, y_test = train_test_split(x_transform, y, test_size=0.3)
		logreg.train(x_train, y_train, 0.01, 500)
		pred = logreg.predict(x_test)
		right = len(pred[pred == y_test])
		print('Accuracy:', right/len(y_test))
	else:
		logreg.train(x_transform, y, 0.01, 500)

	# Save the model
	model = (logreg.save(), scaler.save())
	save_model(model)
