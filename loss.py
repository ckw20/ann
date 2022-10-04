from __future__ import division
import numpy as np


class EuclideanLoss(object):
	def __init__(self, name):
		self.name = name

	def forward(self, input, target):
		# TODO START
		delta = input - target
		return np.sum(delta * delta) / (2 * len(input))
		# TODO NED

	def backward(self, input, target):
		# TODO START
		delta = input - target
		return delta / len(input)
		# TODO NED
		


class SoftmaxCrossEntropyLoss(object):
	def __init__(self, name):
		self.name = name
		# NEW START
		self.h = None
		# NEW END

	def forward(self, input, target):
		# TODO START
		e = np.exp(input)
		s = np.matmul(e, np.ones((e.shape[1], 1)))
		self.h = e / s
		E = - target * np.log(self.h)
		return np.sum(E) / len(input)
		# TODO NED

	def backward(self, input, target):
		# TODO START
		return (self.h - target) / len(input)
		# TODO NED


class HingeLoss(object):
	def __init__(self, name, margin=5):
		self.name = name
		self.margin = margin
		self.E = None
		self.res = None

	def forward(self, input, target):
		# TODO START 
		self.res = np.matmul(input * target, np.ones((len(input[0]), 1)))
		self.res = self.margin - self.res + input
		self.E = np.maximum(0, self.res) * (1 - target)
		return np.sum(self.E) / len(input)
		# TODO END

	def backward(self, input, target):
		# TODO START
		A = np.matmul(self.E > 0, np.ones((len(input[0]), 1))) * target
		return ((self.res >= 0) - target - A) / len(input)
		# TODO END

# NEW START
def test_loss(loss):
	inp = np.linspace(-5, 5, 100).reshape((10, 10))
	tar = np.eye(10)
	val = loss.forward(inp, tar)
	grad = loss.backward(inp, tar)
	dx = 0.01
	mx = 0
	mn = 2
	for i in range(10):
		for j in range(10):
			inp[i][j] += dx
			dy = loss.forward(inp, tar) - val
			inp[i][j] -= dx
			f = dy / dx
			g = grad[i][j]
			mx = max(mx, f / g)
			mn = min(mn, f / g)
	print("testing loss : %s" % loss.name, mn, mx)
# NEW END
