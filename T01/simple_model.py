import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
from sklearn.preprocessing import OneHotEncoder


(xt,yt), (xv,yv) = load_data()
enc = OneHotEncoder(categories='auto')
enc.fit(yt)

yt = enc.transform(yt).toarray()
yv = enc.transform(yv).toarray()

train_items = xt.shape[0]
test_items  = xv.shape[0]

logsPath = "./Graph"

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="Features")
y = tf.placeholder(tf.float32, [None, 10], name="Labels")

model = tf.layers.Conv2D(64,[1,1],padding="same",activation=tf.nn.relu, name="conv1")(x)
model = tf.layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), name="pool1")(model)
model = tf.nn.local_response_normalization(model, name="norm1")
model = tf.layers.Conv2D(64,[5,5],padding="same",activation=tf.nn.relu, name="conv2")(model)
model = tf.nn.local_response_normalization(model, name="norm2")
model = tf.layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), name="2_Pool")(model)

model = tf.layers.Flatten(name="1_Flatten")(model)
model = tf.layers.Dense(2048, activation=tf.nn.relu, name="1_dense")(model)
model = tf.layers.Dense(10, activation=tf.nn.relu, name="2_dense")(model)

# loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=model))
tf.summary.histogram("Loss", cost)

# optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='accuracy')
tf.summary.histogram("Accuracy", accuracy)

epochs = 50
batchSize = 8
n_batches = train_items // batchSize
saver = tf.train.Saver()

histT = []
histV = []
print("Begin Session")
with tf.Session() as sess:
  # if tf.gfile.Exists(logsPath+"model.ckpt"):
  #   saver.restore(sess, logsPath+"model.ckpt")
  #   print("Model restored!")
  train_writer = tf.summary.FileWriter(logsPath, sess.graph)
  sess.run(tf.global_variables_initializer())
  print("Variables Initialized")

  counter = 0
  for epoch in range(epochs):
    print("Start Epoch {:>2}".format(epoch + 1))
    for i in range(n_batches):
      print(".", end="")
      counter += 1
      merge = tf.summary.merge_all()
      summary, _ = sess.run([merge, optimizer],
                            feed_dict={x: xt[i*batchSize:(i+1)*batchSize],
                                       y: yt[i*batchSize:(i+1)*batchSize]})
      train_writer.add_summary(summary, counter)
    print("\nEpoch {:>2}:  ".format(epoch + 1), end='\n')
    vloss, vacc = sess.run([cost,accuracy],
                            feed_dict={x: xv,
                                       y: yv})
    tloss, tacc = sess.run([cost,accuracy],
                             feed_dict={x: xt,
                                        y: yt})
    print("Validation Loss: {:>8.4f}, Validation Accuracy: {:>8.6f}".format(vloss, vacc))
    print("Training   Loss: {:>8.4f}, Training   Accuracy: {:>8.6f}".format(tloss, tacc))
    histT.append(tacc)
    histV.append(vacc)

    # Save Model
  train_writer.flush()
  save_path = saver.save(sess, './image_classification')
