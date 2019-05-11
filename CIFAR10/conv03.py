import tensorflow as tf
from tensorflow.keras.datasets.cifar10 import load_data
from sklearn.preprocessing import OneHotEncoder
from time import time
import numpy as np

(xt,yt), (xv,yv) = load_data()
enc = OneHotEncoder(categories='auto')
enc.fit(yt)

yt = enc.transform(yt).toarray()
yv = enc.transform(yv).toarray()

train_items = xt.shape[0]
test_items  = xv.shape[0]

logsPath = "./Graph03"

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="Features")
y = tf.placeholder(tf.float32, [None, 10], name="Labels")

model = tf.layers.Conv2D(96,[8,8],padding="same",activation=tf.nn.relu, name="1_Conv")(x)
model = tf.layers.MaxPooling2D(pool_size=(3, 3), strides=2, name="1_Pool")(model)
model = tf.layers.BatchNormalization(name="1_batchNormalization")(model)

model = tf.layers.Conv2D(256,[4,4],padding="valid",activation=tf.nn.relu, name="2_Conv")(model)
model = tf.layers.MaxPooling2D(pool_size=(3, 3), strides=2, name="2_Pool")(model)
model = tf.layers.BatchNormalization(name="2_batchNormalization")(model)

model = tf.layers.Conv2D(384,[3,3],padding="valid",activation=tf.nn.relu, name="3_Conv")(model)
model = tf.layers.Conv2D(384,[3,3],padding="valid",activation=tf.nn.relu, name="4_Conv")(model)

model = tf.layers.Conv2D(256,[3,3],padding="same",activation=tf.nn.relu, name="5_Conv")(x)
model = tf.layers.MaxPooling2D(pool_size=(3, 3), strides=2, name="3_Pool")(model)

model = tf.layers.Flatten(name="1_Flatten")(model)

model = tf.layers.Dense(4096, activation=tf.nn.relu, name="1_dense")(model)
model = tf.layers.Dropout(0.2, name="2_Dropout")(model)

model = tf.layers.Dense(4096, activation=tf.nn.relu, name="1_dense")(model)
model = tf.layers.Dropout(0.2, name="2_Dropout")(model)

model = tf.layers.Dense(1000, activation=tf.nn.relu, name="1_dense")(model)

model = tf.layers.Dense(10, name="2_dense")(model)

# loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=model), name="loss")
tf.summary.scalar("Loss", cost)

# optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name="accuracy")
tf.summary.scalar("Accuracy", accuracy)
nskip = 5
epochs = 50
batchSize = 250
n_batches = train_items // batchSize
saver = tf.train.Saver()
print("Begin Session")
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(logsPath, sess.graph)
    sess.run(tf.global_variables_initializer())
    counter = 0
    for epoch in range(epochs):
        print("Start Epoch {:>2}".format(epoch+1))
        t = time()
        for i in range(n_batches):
            sess.run(optimizer,
                     feed_dict={x: xt[i*batchSize:(i+1)*batchSize],
                                y: yt[i*batchSize:(i+1)*batchSize]})
            if i % nskip == 0:
                counter += 1
                merge = tf.summary.merge_all()
                iv = np.random.randint(0, test_items, batchSize)
                summary = sess.run(merge,
                                   feed_dict={x: xv[iv], y: yv[iv]})
                train_writer.add_summary(summary, counter)


        print("Epoch {:>2}:  ".format(epoch + 1), end='\n')
        it = np.random.randint(0, train_items, batchSize)
        iv = np.random.randint(0, test_items, batchSize)
        vloss, vacc = sess.run([cost, accuracy], feed_dict={x: xv[iv], y: yv[iv]})
        tloss, tacc = sess.run([cost, accuracy], feed_dict={x: xt[it], y: yt[it]})
        print("Validation Loss: {:>8.4f}, Validation Accuracy: {:>8.6f}".format(vloss, vacc))
        print("Training   Loss: {:>8.4f}, Training   Accuracy: {:>8.6f}".format(tloss, tacc))
        print("Epoch time: ", time() - t)
