"""
This network feeds the output of a linear transform
to the sigmoid function.

Finish implementing the Sigmoid class in miniflow.py!

Feel free to play around with this network, too!
"""

import numpy as np
from miniflow import *

inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)
g = Sigmoid(f)

x = np.array([[-1., -2.], [-1, -2]])
w = np.array([[2., -3], [2., -3]])
b = np.array([-3., -5])

feed_dict = {inputs: x, weights: w, bias: b}

graph = topological_sort(feed_dict)
output = forward_pass(g, graph)

"""
Output should be:
[[  1.23394576e-04   9.82013790e-01]
 [  1.23394576e-04   9.82013790e-01]]
"""
print(output)


"""
Test MSE method 

"""

y, a = Input(), Input()
cost = MSE(y, a) 

y_ = np.array([1, 2, 3])
a_ = np.array([4.5, 5, 10])

feed_dict = {y: y_, a: a_}
graph = topological_sort(feed_dict)
# forward pass
forward_pass(graph)

"""
Expected output

23.4166666667
"""
print(cost.value)
