#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  logreg.py                                                                   #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Wednesday December 1969 4:00:00 pm                                 #
#  Modified: Wednesday Sep 2019 7:56:05 pm                                     #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import numpy as np

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
