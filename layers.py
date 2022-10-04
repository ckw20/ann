import numpy as np


class Layer(object):
	def __init__(self, name, trainable=False):
		self.name = name
		self.trainable = trainable
		self._saved_tensor = None

	def forward(self, input):
		pass

	def backward(self, grad_output):
		pass

	def update(self, config):
		pass

	def _saved_for_backward(self, tensor):
		'''The intermediate results computed during forward stage
		can be saved and reused for backward, for saving computation'''

		self._saved_tensor = tensor

class Relu(Layer):
	def __init__(self, name):
		super(Relu, self).__init__(name)
	
	def forward(self, input):
		# TODO START
		self._saved_tensor = input > 0
		return np.maximum(input, 0)
		# TODO END

	def backward(self, grad_output):
		# TODO START
		return self._saved_tensor * grad_output
		# TODO END


# NEW START
class LeakyRelu(Layer):
	def __init__(self, name, a=0.0):
		super(LeakyRelu, self).__init__(name)
		self.a = a

	def forward(self, input):
		gt0 = input > 0
		self._saved_tensor = gt0 + (1 - gt0) * self.a
		return self._saved_tensor * input

	def backward(self, grad_output):
		return self._saved_tensor * grad_output;
#NEW END

class Sigmoid(Layer):
	def __init__(self, name):
		super(Sigmoid, self).__init__(name)

	def forward(self, input):
		# TODO START
		self._saved_tensor = 1.0 / (1.0 + np.exp(-input))
		return self._saved_tensor
		# TODO END

	def backward(self, grad_output):
		# TODO START
		return grad_output * self._saved_tensor * (1 - self._saved_tensor)
		# TODO END

class Gelu(Layer):
	def __init__(self, name):
		super(Gelu, self).__init__(name)
		#NEW START
		self.x1 = None
		self.x3 = None
		#NEW END

	def forward(self, input):
		# TODO START
		self.x1 = input
		self.x3 = input * input * input
		x1 = self.x1
		x3 = self.x3
		return 0.5 * x1 * (1 + np.tanh(0.797885 * (x1 + 0.044715 * x3)))
		# TODO END

	def backward(self, grad_output):
		# TODO START
		# d/dx(1/2 x (1 + tanh(sqrt(2/pi) (x + 0.44175 x^3)))) = 1/2 (tanh(0.352466 x^3 + 0.797885 x) + (1.0574 x^3 + 0.797885 x) sech^2(0.352466 x^3 + 0.797885 x) + 1)
		x1 = self.x1
		x3 = self.x3
		v = 0.352466 * x3 + 0.797885 * x1
		s = 2 / (np.exp(x1) + np.exp(-x1))
		return 0.5 * ( np.tanh(v) + (1.0574 * x3 + 0.797885 * x1) * s * s + 1 ) * grad_output
		# TODO END

# NEW START
class Softmax(Layer):
	def __init(self, name):
		super(Softmax, self).__init__(name)

	def forward(self, input):
		e = np.exp(input)
		s = np.matmul(e, np.ones((e.shape[1], 1)))
		self._saved_tensor = e / s
		return self._saved_tensor
	
	def backward(self, grad_output):
		y = self._saved_tensor
		n = y.shape[1]
		return y * (grad_output - np.matmul(grad_output * y, np.ones((n, n))))
# NEW END

class Linear(Layer):
	# MODIFIED START
	def __init__(self, name, in_num, out_num, init_std, alpha = 0.1, beta_m = 0.9, beta_v = 0.99, epsilon = 1):
	# MODIFIED END
		super(Linear, self).__init__(name, trainable=True)
		self.in_num = in_num
		self.out_num = out_num
		self.W = np.random.randn(in_num, out_num) * init_std
		self.b = np.zeros(out_num)

		self.grad_W = np.zeros((in_num, out_num))
		self.grad_b = np.zeros(out_num)

		self.diff_W = np.zeros((in_num, out_num))
		self.diff_b = np.zeros(out_num)

		# NEW START
		self.alpha = alpha
		self.beta_m = beta_m
		self.beta_v = beta_v
		self.epsilon = epsilon
		self.m_W = np.zeros((in_num, out_num))
		self.v_W = np.zeros((in_num, out_num))
		self.m_b = np.zeros(out_num)
		self.v_b = np.zeros(out_num)
		self.beta_m_t = 1
		self.beta_v_t = 1
		# NEW END

	def forward(self, input):
		# TODO START
		self._saved_for_backward(input)
		output = np.matmul(input, self.W) + self.b
		return output
		# TODO END

	def backward(self, grad_output):
		# TODO START
		self.grad_W = np.matmul(self._saved_tensor.T, grad_output)
		self.grad_b = np.matmul(np.ones((1, len(grad_output))), grad_output)
		return np.matmul(grad_output, self.W.T)
		# TODO END

	def update(self, config):
		mm = config['momentum']
		lr = config['learning_rate']
		wd = config['weight_decay']

		# MODIFIED START
		use_adam = 0
		if not use_adam:
			self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
			self.W = self.W - lr * self.diff_W

			self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
			self.b = self.b - lr * self.diff_b
		else:
			warm_up = 0
			use_RMSProp = 0
			if warm_up:
				self.beta_m_t = self.beta_m_t * self.beta_m
				self.beta_v_t = self.beta_v_t * self.beta_v

				self.m_W = self.beta_m * self.m_W + (1 - self.beta_m) * self.grad_W
				self.v_W = self.beta_v * self.v_W + (1 - self.beta_v) * self.grad_W * self.grad_W
				m_W_hat = self.m_W / (1 - self.beta_m_t)
				v_W_hat = self.v_W / (1 - self.beta_v_t)
				self.W = self.W - self.alpha * m_W_hat / np.sqrt(v_W_hat + self.epsilon)
				
				self.m_b = self.beta_m * self.m_b + (1 - self.beta_m) * self.grad_b
				self.v_b = self.beta_v * self.v_b + (1 - self.beta_v) * self.grad_b * self.grad_b
				m_b_hat = self.m_b / (1 - self.beta_m_t)
				v_b_hat = self.v_b / (1 - self.beta_v_t)
				self.b = self.b - self.alpha * m_b_hat / np.sqrt(v_b_hat + self.epsilon)
			elif use_RMSProp:
				self.v_W = self.beta_v * self.v_W + (1 - self.beta_v) * self.grad_W * self.grad_W
				self.v_b = self.beta_v * self.v_b + (1 - self.beta_v) * self.grad_b * self.grad_b
				self.W = self.W - self.alpha * self.grad_W / np.sqrt(self.v_W + self.epsilon)
				self.b = self.b - self.alpha * self.grad_b / np.sqrt(self.v_b + self.epsilon)
			else:
				self.v_W = self.v_W + self.grad_W * self.grad_W
				self.v_b = self.v_b + self.grad_b * self.grad_b
				self.W = self.W - self.alpha * self.grad_W / np.sqrt(self.v_W + self.epsilon)
				self.b = self.b - self.alpha * self.grad_b / np.sqrt(self.v_b + self.epsilon)
		# MODIFIED END

#NEW START
def test_layer(layer):
	n = 2
	m = 2
	inp = np.linspace(- n * m / 10 + 1, n * m, n * m).reshape((n, m))
	output = layer.forward(inp)
	grad = layer.backward(np.ones((n, m)))
	dx = 0.01
	mx = 0
	mn = 2
	for i in range(n):
		for j in range(m):
			inp[i][j] += dx
			dy = layer.forward(inp) - output
			inp[i][j] -= dx
			f = dy[i][j]
			g = grad[i][j] * dx
			mx = max(mx, np.max(np.abs(f - g)))
			mn = min(mn, np.min(np.abs(f - g)))
	print("testing layer : %s " % layer.name, mn, mx)
# NEW END
