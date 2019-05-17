import os
import tensorflow as tf
import numpy as np
from time import time

def motionLoad(classes, directory, window = 60):
    if directory.endswith("/"):
        dire = directory
    else:
        dire = directory+"/"
    file_list = os.listdir(dire)
    file_list = [i for i in file_list if any(s in i for s in classes)]
    instances = []
    labels = []
    #READ FILES
    for f in file_list:
        label = [0.,0.,0.]
        index = 0
        #CHECK CLASS
        for c, cla in enumerate(classes):
            if cla in f:
                label[c] = 1.0
        #OPEN FILE
        with open(directory+f) as doc:
            lines = doc.readlines()
        end = 0
        #READ AND APPEND
        while index < len(lines):
            count = 0
            subin = []
        if index+window < len(lines):
            while count < window:
                count +=1
                out = lines[index].split("|m ")[1].split(" |class")[0].split(" ") # if you make changes look here
                out = [float(i) for i in out]
                subin.append(out)
                index +=1 # pay attention here!
            instances.append(np.array(subin))
            labels.append   (np.array(label))
        else:
            index = len(lines)
  return instances, labels

classes = ["Crouching", "Idle", "Running", "Siwing", "Walking"]
actions = ["Crouching", "Running", "Swing"]
window  = 40
n_actions = len(actions)

xv, yv = motionLoad(actions, "./val/", window)
xt, yt = motionLoad(["Crouching", "Running", "Swing"], "./tr/", window)

x = tf.placeholder(tf.float32, [None, window, 20], name="Inputs")
y = tf.placeholder(tf.float32, [None, n_actions], name="Labels")

model = tf.layers.Conv1D(filters=128, kernel_size=8, name="Conv1")(x)
model = tf.layers.BatchNormalization()(model)
model = tf.nn.relu(model)

model = tf.layers.Conv1D(filters=256, kernel_size=5, name="Conv2")(x)
model = tf.layers.BatchNormalization()(model)
model = tf.nn.relu(model)

model = tf.layers.Conv1D(filters=128, kernel_size=3, name="Conv3")(x)
model = tf.layers.BatchNormalization()(model)
model = tf.nn.relu(model)

model = tf.keras.layers.GlobalAveragePooling1D()(model)
model = tf.layers.Dense(n_actions, name="dense2")(model)


# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=model), name="loss")

# optimizer
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Accuracy
prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name="accuracy")

epochs = 20
batch_size = 20
batches = n_tr // batch_size
print("Begin Session")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    counter = 0
    tic = time()
    for epoch in range(epochs):
        print("Start Epoch {:>2}".format(epoch+1))
        t = time()
        for i in range(batches):
            tloss, tacc, _ = sess.run([loss, accuracy, optimizer],
                                         feed_dict={
                                             x: xt[i*batch_size:(i+1)*batch_size],
                                             y: yt[i*batch_size:(i+1)*batch_size]})
        print("Training   Loss: {:>8.4f}, Training   Accuracy: {:>8.6f}".format(tloss, tacc))
        print("Epoch time: ", time() - t)
    vloss, vacc = sess.run([loss, accuracy], feed_dict={x: xv, y: yv})
    print("Validation Loss: {:>8.4f}, Validation Accuracy: {:>8.6f}".format(vloss, vacc))
    print("Training time:", time() - tic)
