import tensorflow as tf
import numpy as np 
import math
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784], name='X')
Y = tf.placeholder(tf.float32, [None, 10], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
global_step = tf.Variable(0, trainable=False, name='global_step')

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
    B1 = tf.Variable(tf.zeros([256]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + B1)
    L1 = tf.nn.dropout(L1, keep_prob)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
    B2 = tf.Variable(tf.zeros([256]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + B2)
    L2 = tf.nn.dropout(L1, keep_prob)

with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
    B3 = tf.Variable(tf.zeros([10]))
    model = tf.matmul(L2, W3) + B3

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost, global_step=global_step)
    tf.summary.scalar('cost', cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

batch_size = 128
epoches = 30
iter_per_epoch = int(math.ceil(mnist.train.num_examples / batch_size))

for epoch in range(1, epoches + 1):
    total_cost = 0
    for i in range(iter_per_epoch):
        x_batch, y_batch = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost], feed_dict = {
            X: x_batch, Y: y_batch, keep_prob: 0.8
        })
        total_cost += cost_val
        summary = sess.run(merged, feed_dict={X:x_batch, Y:y_batch, keep_prob: 0.8})
        writer.add_summary(summary, global_step=sess.run(global_step))

    print('Epoch:', '%04d' % (epoch), 'Avg.cost = ', '{:.3f}'.format(total_cost / iter_per_epoch))

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("Accuracy=", sess.run(accuracy, feed_dict={
        X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1
    }))

