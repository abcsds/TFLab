#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from time import time


train_items = 10000
test_items  = 100

xt = np.random.rand(train_items, 10)
xv = np.random.rand(test_items, 10)
yt = np.sin(np.sum(xt, axis=1)).reshape(-1, 1)
yv = np.sin(np.sum(xv, axis=1)).reshape(-1, 1)

logsPath = "./GraphDense"

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 10], name="Features")
y = tf.placeholder(tf.float32, [None, 1], name="Labels")

model = tf.layers.Dense(2048, activation=tf.nn.relu, name="1_dense")(x)
model = tf.layers.Dense(1024, activation=tf.nn.relu, name="2_dense")(model)
model = tf.layers.Dense(512, activation=tf.nn.relu, name="3_dense")(model)
out   = tf.layers.Dense(1, activation=tf.nn.sigmoid, name="4_dense")(model)

# loss
loss = tf.losses.mean_squared_error(y, out)
tf.summary.scalar("Loss", loss)

# optimizer
optimizer = tf.train.AdamOptimizer().minimize(loss)


epochs = 50
batchSize = train_items
n_batches = train_items // batchSize

saver = tf.train.Saver()
print("Begin Session")
t = time()
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(logsPath, sess.graph)

    sess.run(tf.global_variables_initializer())
    print("Variables Initialized")

    counter = 0
    for epoch in range(epochs):
        print("Start Epoch {:>2}".format(epoch + 1))
        counter += 1
        for i in range(n_batches):
            print(".", end="")
            sess.run(optimizer,
               feed_dict={x: xt[i * batchSize:(i + 1) * batchSize],
                          y: yt[i * batchSize:(i + 1) * batchSize]})

        merge = tf.summary.merge_all()
        summary = sess.run(merge, feed_dict={x: xv, y: yv})
        train_writer.add_summary(summary, counter)
        print("\nEpoch {:>2}:  ".format(epoch + 1), end='\n')
        vloss = sess.run(loss, feed_dict={x: xv, y: yv})
        tloss = sess.run(loss, feed_dict={x: xt, y: yt})
        print("Validation Loss: {:>8.4f}".format(vloss))
        print("Training   Loss: {:>8.4f}".format(tloss))


print("Elapsed time:", time() - t)
#
