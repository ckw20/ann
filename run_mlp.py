from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Gelu, LeakyRelu, Softmax
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, HingeLoss
from loss import test_loss
from layers import test_layer
from solve_net import train_net, test_net
from load_data import load_mnist_2d


train_data, test_data, train_label, test_label = load_mnist_2d('data')

"""
# test layers and losses
test_loss(EuclideanLoss('EuclideanLoss'))
test_loss(SoftmaxCrossEntropyLoss('SoftmaxCrossEntropyLoss'))
test_layer(Softmax('Softmax'))
test_layer(Relu('Relu'))
test_layer(LeakyRelu('LeakyRelu', 0.05))
test_layer(Sigmoid('Sigmoid'))
test_layer(Gelu('Gelu'))
test_loss(HingeLoss('HingeLoss', 2))
"""


"""
# a normal model of one layer
model = Network()
model.add(Linear('fc1', 784, 10, 0.1))
model.add(Sigmoid('Sigmoid'))
loss = EuclideanLoss(name='Euc')
"""


"""
# a model of two linear layers
model = Network()
model.add(Linear('fc1',784,512,0.1))
model.add(Linear('fc2',100,10,0.1))
model.add(Sigmoid('Sigmoid'))
loss = EuclideanLoss(name='Euc')
"""


# a good model
model = Network()
model.add(Linear('fc1', 784, 512, 0.1))
model.add(LeakyRelu('LeakyRelu', 0.05))
model.add(Linear('fc1', 512, 256, 0.1))
model.add(Gelu('Gelu'))
model.add(Linear('fc1', 256, 128, 0.1))
model.add(Sigmoid('Sigmoid'))
model.add(Linear('fc1', 128, 64, 0.1))
model.add(LeakyRelu('LeakyRelu', 0.3))
model.add(Linear('fc1', 64, 10, 0.1))
model.add(LeakyRelu('LeakyRelu', 0.03))
loss = SoftmaxCrossEntropyLoss(name='Soft')


# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#	   one epoch means model has gone through all the training samples.
#	   'disp_freq' denotes number of iterations in one epoch to display information.

config = {
	'learning_rate': 0.28,
	'weight_decay': 0.00,
	'momentum': 0.05,
	'batch_size': 50,
	'max_epoch': 100,
	'disp_freq': 100,
	'test_epoch': 1
}

lr = config['learning_rate']

for epoch in range(config['max_epoch']):
	LOG_INFO('Training @ %d epoch...' % (epoch))
	config['learning_rate'] = lr - lr / config['max_epoch'] * epoch
	print("learning rate =", config['learning_rate'])

	train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
	if epoch > 0 and epoch % config['test_epoch'] == 0:
		LOG_INFO('Testing @ %d epoch...' % (epoch))
		test_net(model, loss, test_data, test_label, config['batch_size'])

#test_net(model, loss, test_data, test_label, config['batch_size'])
