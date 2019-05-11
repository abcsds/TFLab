import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.mnist import load_data
from sklearn.preprocessing import OneHotEncoder
from time import time


(xt,yt), (xv,yv) = load_data()
xt = xt.reshape((xt.shape[0],28,28,-1))
xv = xv.reshape((xv.shape[0],28,28,-1))
enc = OneHotEncoder(categories="auto")
enc.fit(yt.reshape((-1,1)))
yt = enc.transform(yt.reshape((-1,1))).toarray()
yv = enc.transform(yv.reshape((-1,1))).toarray()

train_items = xt.shape[0]
test_items  = xv.shape[0]


logsPath = "./Graph01"

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="Features")
y = tf.placeholder(tf.float32, [None, 10], name="Labels")

model = tf.layers.Conv2D(10,[14,14],padding="same",activation=tf.nn.relu, name="conv1")(x)
model = tf.layers.BatchNormalization(name="batch1")(model)
model = tf.layers.Conv2D(32,[7,7],padding="same",activation=tf.nn.elu, name="conv2")(model)
model = tf.layers.BatchNormalization(name="batch2")(model)
model = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool1")(model)
model = tf.layers.Dropout(0.2, name="drop1")(model)
model = tf.layers.Flatten()(model)
model = tf.layers.Dense(2048, name="dense1")(model)
model = tf.layers.Dense(10, name="dense2")(model)

# loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=model))
tf.summary.scalar("Loss", cost)

# optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='accuracy')
tf.summary.scalar("Accuracy", accuracy)

epochs = 50
batchSize = 10
# batchSize = train_items
n_batches = train_items // batchSize
saver = tf.train.Saver()

histT = []
histV = []
print("Begin Session")
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(logsPath, sess.graph)
    sess.run(tf.global_variables_initializer())
    counter = 0
    for epoch in range(epochs):
        print("Start Epoch {:>2}".format(epoch + 1))
        t = time()
        counter += 1
        for i in range(n_batches):
            print(".", end="")
            sess.run(optimizer,
                     feed_dict={x: xt[i * batchSize:(i + 1) * batchSize],
                                y: yt[i * batchSize:(i + 1) * batchSize]})
        merge = tf.summary.merge_all()
        summary = sess.run(merge, feed_dict={x: xt, y: yt})
        train_writer.add_summary(summary, counter)
        print("\nEpoch {:>2}:  ".format(epoch + 1), end='\n')
        vloss, vacc = sess.run([cost, accuracy], feed_dict={x: xv, y: yv})
        tloss, tacc = sess.run([cost, accuracy], feed_dict={x: xt, y: yt})
        print("Validation Loss: {:>8.4f}, Validation Accuracy: {:>8.6f}".format(vloss, vacc))
        print("Training   Loss: {:>8.4f}, Training   Accuracy: {:>8.6f}".format(tloss, tacc))
        print("Epoch time:", time()-t)
        histT.append(tacc)
        histV.append(vacc)
