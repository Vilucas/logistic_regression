#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  logreg.py                                                                   #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Wednesday December 1969 4:00:00 pm                                 #
#  Modified: Tuesday Sep 2019 9:26:10 pm                                       #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import sys
import sklearn.preprocessing as prp
from sklearn.model_selection import train_test_split

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def predict(theta, X):
	return sigmoid(X @ theta)

def cost(theta, X, y):
	predictions = predict(theta, X)
	error = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
	return np.sum(error) / len(error)

def cost_gradient(theta, X, y):
	predictions = predict(theta, X)
	return X.T @ (predictions - y ) / len(y)

def step_gradient(X, Y, theta, eta):
	# Y_hat = predict(theta, X)
	gradient = cost_gradient(theta, X, Y)
	return theta - gradient * eta

class LogReg:
	def __init__(self):
		self.solved = False
	def train(self, x, y, eta, max_iter):
		X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
		X[:, 1:] = x
		theta = np.zeros(X.shape[1])
		for _ in range(max_iter):
			theta = step_gradient(X, y, theta, eta)
		self.theta = theta
		self.solved = True
	def predict(self, x):
		if not self.solved:
			raise Exception('model has not been trained')
		if x.ndim > 1:
			X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
			X[:, 1:] = x
		else:
			X = np.ones(x.shape[0] + 1)
			X[1:] = x
		return predict(self.theta, X)

if __name__ == '__main__':
	if (len(sys.argv)) != 2:
		exit(0)
	try:
		data = pd.read_csv(sys.argv[1])
	except Exception as e:
		print(e)
		exit(1)

	scalar = prp.StandardScaler()
	logreg = LogReg()
	data = data.dropna()
	y = data['Hogwarts House']
	y.loc[y[:] != 'Slytherin'] = 0
	y.loc[y[:] == 'Slytherin'] = 1
	x = data.select_dtypes(include=[np.number])[['Divination', 'Arithmancy']]
	scalar.fit(x)
	x = scalar.transform(x)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
	logreg.train(x_train, y_train, .1, 500)
	# print(logreg.predict(np.array([0.36, 0.53])))
	# print(logreg.predict(np.array([-2.14, 1.06])))
	# for i in x_test:
	print(pd.DataFrame(y_test))
	out = logreg.predict(x_test)
	print(pd.DataFrame(out))
	out[out[:] < 0.5] = 0
	out[out[:] > 0.5] = 1
	print(pd.DataFrame(out))
	print(len(out))
	print(len(out == y_test))
	# print(y_test)
