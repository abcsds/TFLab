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

logsPath = "./Graph01"

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="Features")
y = tf.placeholder(tf.float32, [None, 10], name="Labels")

model = tf.layers.Conv2D(32,[3,3],padding="same",activation=tf.nn.elu, name="1_Conv")(x)
model = tf.layers.BatchNormalization(name="1_batchNormalization")(model)
model = tf.layers.Conv2D(32,[3,3],padding="same",activation=tf.nn.elu, name="2_Conv")(model)
model = tf.layers.BatchNormalization(name="2_batchNormalization")(model)
model = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=1, name="1_Pool")(model)
model = tf.layers.Dropout(0.2, name="1_Dropout")(model)
model = tf.layers.Conv2D(64,[3,3],padding="same",activation=tf.nn.elu, name="3_Conv")(model)
model = tf.layers.BatchNormalization(name="3_batchNormalization")(model)
model = tf.layers.Conv2D(64,[3,3],padding="same",activation=tf.nn.elu, name="4_Conv")(model)
model = tf.layers.BatchNormalization(name="4_batchNormalization")(model)
model = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=1, name="2_Pool")(model)
model = tf.layers.Dropout(0.2, name="2_Dropout")(model)
model = tf.layers.Conv2D(128,[3,3],padding="same",activation=tf.nn.elu, name="5_Conv")(model)
model = tf.layers.BatchNormalization(name="5_batchNormalization")(model)
model = tf.layers.Conv2D(128,[3,3],padding="same",activation=tf.nn.elu, name="6_Conv")(model)
model = tf.layers.BatchNormalization(name="6_batchNormalization")(model)
model = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=1, name="2_Pool")(model)
model = tf.layers.Dropout(0.2, name="2_Dropout")(model)
model = tf.layers.Flatten(name="1_Flatten")(model)
model = tf.layers.Dense(2048, name="1_dense")(model)
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

epochs = 50
batchSize = 500
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
            sess.run(optimizer, feed_dict={x: xt[i*batchSize:(i+1)*batchSize],
                                           y: yt[i*batchSize:(i+1)*batchSize]})

        print("Epoch {:>2}:  ".format(epoch + 1), end='\n')
        it = np.random.randint(0, train_items, batchSize)
        iv = np.random.randint(0, test_items, batchSize)
        counter += 1
        merge = tf.summary.merge_all()
        summary = sess.run(merge, feed_dict={x: xv[iv], y: yv[iv]})
        train_writer.add_summary(summary, counter)
        vloss, vacc = sess.run([cost, accuracy], feed_dict={x: xv[iv], y: yv[iv]})
        tloss, tacc = sess.run([cost, accuracy], feed_dict={x: xt[it], y: yt[it]})
        print("Validation Loss: {:>8.4f}, Validation Accuracy: {:>8.6f}".format(vloss, vacc))
        print("Training   Loss: {:>8.4f}, Training   Accuracy: {:>8.6f}".format(tloss, tacc))
        print("Epoch time: ", time() - t)
