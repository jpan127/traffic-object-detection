import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# turn off all warnings 
tf.logging.set_verbosity(tf.logging.ERROR)

# turn off warnings about cpu optimizations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load mnist training set
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_classes  = 10
batch_size = 100
s_image	   = 784
n_nodes_hl = [s_image, 500, 500, 500, n_classes]

# Shape: [dim1, dim2] with dim1 being the index in a list, and dim2 being the index for each pixel
# None means the shape can be of any length
x = tf.placeholder('float', [None, 784])	# 28 * 28 = 784
y = tf.placeholder('float')

def NN_model (data):

	nLayers = 4

	hidden_layers = []

	# we pass random_normal() a shape, and this shape is filled with random values
	# 3 hidden layers + 1 output layer
	for i in range(nLayers):
		hidden_layers.append( {'weights': tf.Variable(tf.random_normal([n_nodes_hl[i], n_nodes_hl[i+1]])),
								'biases': tf.Variable(tf.random_normal([n_nodes_hl[i+1]]))} )

	layer = [data]

	pkeep = tf.constant(0.95, dtype=tf.float32)

	# 3 hidden layers
	for i in range(nLayers-1):
		# add the next layer [l1, l2, l3] to the list
		layer.append( tf.add( tf.matmul(layer[i], hidden_layers[i]['weights']), hidden_layers[i]['biases'] ) )
		# apply ReLU on the newly added layer
		layer[i+1] = tf.nn.relu(layer[i+1])
		# dropout layer randomly drops out a percentage to avoid overfitting
		# layer[i+1] = tf.nn.dropout(layer[i+1], pkeep)	# adding dropout 0.9734

	# 1 output layer
	ol = tf.add( tf.matmul(layer[nLayers-1], hidden_layers[nLayers-1]['weights']), hidden_layers[nLayers-1]['biases'] )

	return ol


def train_NN (x, y):
	
	prediction = NN_model(x)
	# our cost function
	# softmax regression is a model that gives us a list of values between [0,1] that add up to 1
	# it normalizes the exponent of its inputs
	# softmax(x) = normalize(exp(x))
	# cross entropy measures how inefficient our predictions are for describing the truth
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	optimizer = tf.train.AdamOptimizer().minimize(cost)						# 100 epochs: 0.9753
	# optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost) 	# 100 epochs: 0.9448

	# number of training cycles
	hm_epochs = 100

	with tf.Session() as session:
		session.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			# number of training steps
			for _ in range(int(mnist.train.num_examples/batch_size)):
				# grab next random batch (stochastic gradient descent)
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = session.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
			print("Epoch", epoch, "completed out of", hm_epochs, "Loss:", epoch_loss)

		# computes if equal
		# argmax returns the index of the highest entry in a tensor along some axis
		# this function returns a list of booleans 
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

		# we need to cast the list of booleans into floats to evaluate a mean (fraction)
		# computes the mean, can be passed a dimension, but not here,
		# so just computes the mean of correct
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		# run the accuracy on our test data
		print("Accuracy:", accuracy.eval( {x: mnist.test.images, y: mnist.test.labels} ))

###############################################################################################################

## 	SUMMARY
#	Create layers with weights and biases
#	Matrix multiply input and weights, then previous weights and current weights
#	Add result to bias
#	Use an activation function, for example ReLU
#	Compound on previous layer until output layer
# 	Create a cost function based on reduce mean of softmax cross entropy with logits of output layer
#	Create an optimizer that trains using AdamOptimizer that minimizes the cost function

## 	Start Program
train_NN(x, y)

# print( tf.random_normal([784, 500]) )
# print( tf.Variable(tf.random_normal([784, 500])) )