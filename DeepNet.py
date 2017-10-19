import numpy as np
import tensorflow as tf

#This imports tensorflow's example mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


sess = tf.InteractiveSession()

#Declare all our variables
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W1 = tf.Variable(tf.random_uniform([784,400], dtype=tf.float32))
b1 = tf.Variable(tf.random_uniform([400], dtype=tf.float32))
W2 = tf.Variable(tf.random_uniform([400,200], dtype=tf.float32))
b2 = tf.Variable(tf.random_uniform([200], dtype=tf.float32))
W3 = tf.Variable(tf.random_uniform([200,100], dtype=tf.float32))
b3 = tf.Variable(tf.random_uniform([100], dtype=tf.float32))
W4 = tf.Variable(tf.random_uniform([100,50], dtype=tf.float32))
b4 = tf.Variable(tf.random_uniform([50], dtype=tf.float32))
W5 = tf.Variable(tf.random_uniform([50,10], dtype=tf.float32))
b5 = tf.Variable(tf.random_uniform([10], dtype=tf.float32))
sess.run(tf.global_variables_initializer())

#Here's the forward propagation step of our neural net.
A1 = tf.nn.relu(tf.matmul(x,W1) + b1)
A2 = tf.nn.relu(tf.matmul(A1, W2) + b2)
A3 = tf.nn.relu(tf.matmul(A2, W3) + b3)
A4 = tf.nn.relu(tf.matmul(A3, W4) + b4)
y = tf.matmul(A4,W5) + b5

#This tells TensorFlow what our cost function is, and how we want to interact with the gradients of our function. 
#TF calculates gradients automatically - we don't have to write a backprop algorithm.
#The second line "cross_entropy +=..." adds L2 regularization
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#cross_entropy += (.03 * (tf.reduce_mean(tf.pow(W1, 2)) + tf.reduce_mean(tf.pow(W2, 2)) + tf.reduce_mean(tf.pow(W3, 2)) + tf.reduce_mean(tf.pow(W5, 2))))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
sess.run(tf.global_variables_initializer())

#Run 10,000 iterations of Adam, printing the test set performance after every 1,000 iterations.
#It took 30,000-40,000 iterations for this neural net to achieve ~97.5% test set accuracy on MNIST.
for i in range(0,10):
    for _ in range(1000):
      batch = mnist.train.next_batch(100)
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    #print(sess.run(cross_entropy, feed_dict = {x: batch[0], y_: batch[1]}))


#To save weights:
#saver = tf.train.Saver()
#saver.save(sess, "/Users/prestonevans/Downloads/TFtemplates/model.ckpt")

#To load a previous save:
#saver.restore(sess, "/Users/prestonevans/Downloads/TFtemplates/model.ckpt")
