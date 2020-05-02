"""
Implement Add and Multiple function

"""

from miniflow import *

x, y, z = Input(), Input(), Input()

f = Add(x, y, z)
m = Mul(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)

# For Add Class 
add_output = forward_pass(f, graph)

# For Mul class
mul_output = forward_pass(m, graph)

# should output 19
print("{} + {} + {} = {} (according to miniflow for addition )".format(feed_dict[x], feed_dict[y], feed_dict[z], add_output))

print("{} + {} + {} = {} (according to miniflow for multiplication)".format(feed_dict[x], feed_dict[y], feed_dict[z], mul_output))


"""
Implement Linear Function

"""

x, y, z = Input(), Input(), Input()
inputs = [x, y, z]

weight_x, weight_y, weight_z = Input(), Input(), Input()
weights = [weight_x, weight_y, weight_z]

bias = Input()

f = Linear(inputs, weights, bias)

feed_dict = {
	x: 6,
	y: 14,
	z: 3,
	weight_x: 0.5,
	weight_y: 0.25,
	weight_z: 1.4,
	bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

print(output) # should be 12.7 with this example
