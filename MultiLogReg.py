#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  MultiLogReg.py                                                              #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Wednesday December 1969 4:00:00 pm                                 #
#  Modified: Wednesday Sep 2019 9:25:17 pm                                     #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt

def one_hot(y):
	return (np.arange(np.max(y) + 1) == y[:, None]).astype(float)

def to_classlabel(z):
	return z.argmax(axis=1)

def net_input(X, W, b):
	return (X.dot(W) + b)

def softmax(z):
	return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

def cross_entropy(output, y_target):
	return -np.sum(np.log(output) * (y_target), axis=1)

def cost(output, y_target):
	return np.mean(cross_entropy(output, y_target))

class MultiLogReg:
	def __init__(self, verbose=False, plot=False):
		self.solved = False
		self.cost = []
		self.verbose = verbose
		self.plot = plot

	def train(self, x, y, eta, max_iter):
		y_enc = one_hot(y)
		num_classes = np.max(y) + 1
		b = np.ones(num_classes)
		w = np.ones([x.shape[1], num_classes])
		if self.verbose:
			print('y_enc', y_enc, sep='\n')
			print('num_classes:',num_classes)
		for i in range(max_iter):
			net = net_input(x, w, b)
			smax = softmax(net)
			xent = cross_entropy(smax, y_enc)
			J_cost = cost(smax, y_enc)
			self.cost.append(J_cost)
			diff = smax - y_enc
			grad = np.dot(x.T, diff)
			w -= eta * grad
			b -= eta *np.sum(diff, axis=0)
			if self.verbose:
				print(f'\nround {i}')
				print('net_in',net, sep='\n')
				print('softmax', smax, sep='\n')
				print('cross_ent', xent, sep='\n')
				print('grad', grad, sep='\n')
				print('j_cost', J_cost, sep='\n')
				print('w', w, sep='\n')
				print('b', b, sep='\n')
		self.w = w
		self.b = b
		self.solved = True
		if self.verbose:
			print('final w', w, sep='\n')
			print('final cost', J_cost, sep='\n')
		if self.plot:
			plt.plot(self.cost)
			plt.title('Cost')
			plt.xlabel('iterations')
			plt.show()


	def predict(self, x):
		if not self.solved:
			raise Exception('model has not been trained')
		return to_classlabel(softmax(net_input(x, self.w, self.b)))

	def save(self):
		return (self.w, self.b)

	def load(self, model):
		self.w = model[0]
		self.b = model[1]
		self.solved = True