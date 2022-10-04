### In layer.py

1. Add a new layer `LeakyRelu`: $f(x) = x$ when $x \le 0$ and $f(x) = a * x$ when $x < 0$, and $a$ is a constant.
2. Add a new layer `Softmax`: $f(x_i) = \frac(e^{x_i}}{\sum_j e^{x_j}}$.
3. In `Gelu`, add two new parameters to save intermediate result(to improve speed).
4. In `Linear`, add three new method to optimize learning_rate(adam, adam with warm up, RMSProp). Additionally, add several parameters in the constuction function reprensent hyperparameters for these optimal algorithms.
5. Add a `test_layer` function to test layer's forward and backward's correctness.

### In loss.py

1. In `SoftmaxCrossEntropyLoss`, add a parameter to save intermediate result.
2. Add a `test_loss` function to test loss's forward and backward's correctness.

### solve_net.py

1. In `train_net` and `test_net`, modify `%.4f` to `%.4f%%`(eg. $0.9812$ => $98.12\%$).
