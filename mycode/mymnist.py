import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
os.listdir('MNIST_data')

with tf.name_scope('vars') as scope:
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    W = tf.Variable(tf.zeros([784, 10]), name='W')
    b = tf.Variable(tf.zeros([10]), name='b')
    y = tf.nn.softmax(tf.matmul(x, W) + b, name='y')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='conv2d')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name='mp22')

with tf.name_scope('layer1') as scope:
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('layer2') as scope:
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('denselayer') as scope:
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.name_scope('dropout') as scope:
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('readout') as scope:
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.name_scope('evaluation') as scope:
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv), name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    my_writer = tf.summary.FileWriter('/home/jupyter/tlog/cnn/', sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, train_accuracy = sess.run([merged, accuracy], feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0}, options=run_options, run_metadata=run_metadata)
            print('step %d, training accuracy %g' % (i, train_accuracy))
            my_writer.add_run_metadata(run_metadata, 'step%d' % i)
            my_writer.add_summary(summary, i)

        else:
            summary, _ = sess.run([merged, train_step],
                                  feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            my_writer.add_summary(summary, i)

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))