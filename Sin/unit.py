#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from time import time

train_items = 19000
test_items  = 100

xt = np.random.rand(train_items,1)
xv = np.random.rand(train_items,1)
yt = np.sin(xt).reshape(-1,1)
yv = np.sin(xv).reshape(-1,1)

logsPath = "./GraphUnit"

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None,1], name="Features")
y = tf.placeholder(tf.float32, [None,1], name="Labels")

model = tf.layers.Dense(256, activation=tf.nn.relu, name="1_dense")(x)
model = tf.layers.Dense(128, activation=tf.nn.relu, name="2_dense")(model)
out   = tf.layers.Dense(1, activation=tf.nn.sigmoid, name="3_dense")(model)

# loss
loss = tf.losses.mean_squared_error(y, out)

# optimizer
optimizer = tf.train.AdamOptimizer().minimize(loss)

epochs = 10
batchSize = train_items
n_batches = train_items // batchSize

print("Begin Session")
with tf.Session() as sess:
  train_writer = tf.summary.FileWriter(logsPath, sess.graph)
  sess.run(tf.global_variables_initializer())
  print("Variables Initialized")

  counter = 0
  for epoch in range(epochs):
    print("Start Epoch {:>2}".format(epoch + 1))
    t = time()
    for i in range(n_batches):
      print(".", end="")
      sess.run(optimizer, feed_dict={x: xt[i*batchSize:(i+1)*batchSize],
                                     y: yt[i*batchSize:(i+1)*batchSize]})
    merge = tf.summary.merge_all()
    print("\nEpoch {:>2}:  ".format(epoch + 1), end='\n')
    vloss = sess.run(loss, feed_dict={x: xv, y: yv})
    tloss = sess.run(loss, feed_dict={x: xt, y: yt})
    print("Validation Loss: {:>8.4f}".format(vloss))
    print("Training   Loss: {:>8.4f}".format(tloss))

    print("Epoch time:", time()-t)
#
