import math
import numpy as np 
import h5py
import pandas as pd
import matplotlib.pyplot as plt 
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 


#helping functions
def create_placeholders(n_h , n_w , n_c , n_y):
	X = tf.placeholder("float" , [None , n_h , n_w , n_c] , name ='X')
	Y = tf.placeholder("float" , [None , n_y])
	return X , Y

def initialize_parameters():
	#[a,b,c,d] -- a , b - filter size ,  c - input channels , d-output channels
	W1 = tf.get_variable("W1" , [5 ,5 , 1, 32]  , initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W2 = tf.get_variable("W2" , [5 ,5 , 32 , 64] , initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	parameters = {"W1":W1 , "W2":W2}
	return parameters 

def convert_to_one_hot(labels , C):
	C = tf.constant(C , name = 'C')
	one_hot_matrix = tf.one_hot(labels , C , axis = 0)
	sess = tf.Session()
	one_hot = sess.run(one_hot_matrix)
	sess.close()
	return one_hot

def forward_propagation(X , parameters):
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	# k size - window size
	Z1 = tf.nn.conv2d(X , W1 , strides = [1,1,1,1] , padding = 'SAME' )
	A1 = tf.nn.relu(Z1)
	P1 = tf.nn.max_pool(A1 , ksize = [1,8,8,1] , strides = [1,8,8,1] , padding = 'SAME')
	Z2 = tf.nn.conv2d(P1 , W2 , strides = [1,1,1,1] , padding ='SAME')
	A2 = tf.nn.relu(Z2)
	P2 = tf.nn.max_pool(A2 , ksize = [1,4,4,1] , strides = [1,4,4,1] , padding = 'SAME')
	P2 = tf.contrib.layers.flatten(P2)
	Z3 = tf.contrib.layers.fully_connected(P2 , num_outputs = 10 , activation_fn = None)

	return Z3

def compute_cost(Z3 , Y):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3 , labels = Y))
	return cost

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    m = X.shape[0]                  
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def model(X_train , Y_train , X_test , Y_test , learning_rate , num_epochs , minibatch_size):

	tf.set_random_seed(1)
	seed = 3
	(m , n_h , n_w  , n_c) = X_train.shape
	n_y = Y_train.shape[1]

	costs = []

	parameters = initialize_parameters()
	X , Y = create_placeholders(n_h , n_w , n_c , n_y)
	Z3 = forward_propagation(X , parameters)
	cost = compute_cost(Z3 , Y)
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:

		sess.run(init)

		for epoch in range(num_epochs):
			epoch_cost = 0
			num_minibatches = int(m / minibatch_size)
			seed = seed + 1
			minibatches = random_mini_batches(X_train , Y_train , minibatch_size , seed)
			for minibatch in minibatches:
				X_minibatch , Y_minibatch = minibatch
				_ , minibatch_cost = sess.run([optimizer , cost] , feed_dict = {X:X_minibatch , Y:Y_minibatch})
				epoch_cost += minibatch_cost / num_minibatches

			if epoch % 5 == 0:
				costs.append(epoch_cost)
				print("cost after {} epochs: {}".format(epoch , epoch_cost))

		#plot 
		plt.plot(np.squeeze(costs))
		plt.xlabel('cost')
		plt.ylabel('epochs(per 2)')
		plt.title('learning rate'+str(learning_rate))
		plt.show()

		prediction = tf.equal(tf.argmax(Z3,1) , tf.argmax(Y,1))
		accuracy = tf.reduce_mean(tf.cast(prediction , 'float'))

		print("Train Accuracy:", accuracy.eval({X:X_train , Y:Y_train}))
		print("Test Accuracy:" , accuracy.eval({X:X_test , Y:Y_test}))

		parameters = sess.run(parameters)
		return parameters


#creating training and cross validation data
df = pd.read_csv("./train.csv")
df = shuffle(df)

X = np.array(df.drop(['label'] , axis = 1))
Y = np.array(df['label'])

X = X.reshape((-1 , 28 , 28 , 1))

X_train_orig , X_test_orig , Y_train_orig , Y_test_orig = train_test_split(X , Y , test_size = 0.2)

X_train = X_train_orig / 255
X_test  = X_test_orig / 255

Y_train = convert_to_one_hot(Y_train_orig , 10).T 
Y_test  = convert_to_one_hot(Y_test_orig  , 10).T


parameters = model(X_train , Y_train , X_test , Y_test , learning_rate = 0.0001 , num_epochs = 20 ,minibatch_size = 100 )


