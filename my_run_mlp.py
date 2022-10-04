from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Gelu, LeakyRelu, Softmax
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, HingeLoss
from loss import test_loss
from layers import test_layer
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import matplotlib.pyplot as plt
import numpy as np
import ast
import sys
"""
# test layers and losses
test_loss(EuclideanLoss('EuclideanLoss'))
test_loss(SoftmaxCrossEntropyLoss('SoftmaxCrossEntropyLoss'))
test_loss(HingeLoss('HingeLoss', 2))
test_layer(Softmax('Softmax'))
test_layer(Relu('Relu'))
test_layer(LeakyRelu('LeakyRelu', 0.05))
test_layer(Sigmoid('Sigmoid'))
test_layer(Gelu('Gelu'))
"""

config = {
	'learning_rate': 0.28,
	'weight_decay': 0.00,
	'momentum': 0.05,
	'batch_size': 50,
	'max_epoch': 100,
	'disp_freq': 100,
	'test_epoch': 1
}

if __name__ == "__main__":
	config = ast.literal_eval(input())
	print('config =', config)

	model = Network()
	train_data, test_data, train_label, test_label = load_mnist_2d('data')
	
	N = int(input())
	for i in range(0, N):
		r = input().split();
		in_num = int(r[0])
		out_num = int(r[1])
		layer = r[2];
		model.add(Linear('fc', in_num, out_num, 0.1))
		sys.stdout.write("layer %d : in_num = %d out_num = %d " % (i, in_num, out_num))

		if layer == 'Relu':
			model.add(Relu('Relu'))
			print('Relu')
		elif layer == 'LeakyRelu':
			model.add(LeakyRelu('LeakyRelu', float(r[3])))
			print('LeakuRelu(a = %f)' % (float(r[3])))
		elif layer == 'Gelu':
			model.add(Gelu('Gelu'))
			print('Gelu')
		elif layer == 'Sigmoid':
			print('Sigmoid')
			model.add(Sigmoid('Sigmoid'))
		else:
			print('Softmax')
			model.add(Softmax('Softmax'))

	loss_func = input()
	if loss_func == 'Euclidean':
		print('loss_function = EuclideanLoss')
		loss = EuclideanLoss('EuclideanLoss')
	elif loss_func == 'SoftmaxCrossEntropy':
		print('loss_function = SoftmaxCrossEntropy')
		loss = SoftmaxCrossEntropyLoss('SoftmaxCrossEntropyLoss')
	else:
		a = float(loss_func.split()[1])
		print('loss_function = HingeLoss(a = %.5f)' % (a))
		loss = HingeLoss('HingeLoss', a)

	lr = config['learning_rate']
	train_loss = []
	test_loss = []
	train_acc = []
	test_acc = []
	for epoch in range(config['max_epoch']):
		LOG_INFO('Training @ %d epoch...' % (epoch))
		config['learning_rate'] = lr - lr / config['max_epoch'] * epoch
		LOG_INFO("learning rate = %f" % (config['learning_rate']))
		epoch_loss1, epoch_acc1 = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
		train_loss.append(epoch_loss1)
		train_acc.append(epoch_acc1)
		LOG_INFO('Testing @ %d epoch...' % (epoch))
		epoch_loss2, epoch_acc2 = test_net(model, loss, test_data, test_label, config['batch_size'])
		test_loss.append(epoch_loss2)
		test_acc.append(epoch_acc2)

	print('train_loss : %.5f train_acc : %.5f%%\ntest_loss : %.5f train_acc : %.5f%%' % (epoch_loss1, epoch_acc1, epoch_loss2, epoch_acc2))

	X = np.arange(0, config['max_epoch'], 1)
	plt.subplot(1, 2, 1)
	plt.plot(X, np.asarray(train_loss), X, np.asarray(test_loss))
	plt.legend(['train','test'])
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.title('train and test loss')
	
	plt.subplot(1, 2, 2)
	plt.plot(X, np.asarray(train_acc), X, np.asarray(test_acc))
	plt.legend(['train','test'])
	plt.xlabel('epoch')
	plt.ylabel('acc')
	plt.title('train and test acc')
	plt.savefig(input())

