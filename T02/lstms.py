import tensorflow as tf
import numpy as np
import urllib
from time import time

files = ["Actor0Crouching.txt", "Actor0Idle.txt", "Actor0Running.txt", "Actor0Swing.txt", "Actor0Walking.txt", "Actor10Crouching.txt", "Actor10Idle.txt", "Actor10Running.txt", "Actor10Swing.txt", "Actor10Walking.txt", "Actor1Crouching.txt", "Actor1Idle.txt", "Actor1Running.txt", "Actor1Swing.txt", "Actor1Walking.txt", "Actor2Crouching.txt", "Actor2Idle.txt", "Actor2Running.txt", "Actor2Swing.txt", "Actor2Walking.txt", "Actor3Crouching.txt", "Actor3Idle.txt", "Actor3Running.txt", "Actor3Swing.txt", "Actor3Walking.txt", "Actor4Crouching.txt", "Actor4Idle.txt", "Actor4Running.txt", "Actor4Swing.txt", "Actor4Walking.txt", "Actor5Crouching.txt", "Actor5Idle.txt", "Actor5Running.txt", "Actor5Swing.txt", "Actor5Walking.txt", "Actor6Crouching.txt", "Actor6Idle.txt", "Actor6Running.txt", "Actor6Swing.txt", "Actor6Walking.txt", "Actor7Crouching.txt", "Actor7Idle.txt", "Actor7Running.txt", "Actor7Swing.txt", "Actor7Walking.txt", "Actor8Crouching.txt", "Actor8Idle.txt", "Actor8Running.txt", "Actor8Swing.txt", "Actor8Walking.txt", "Actor9Crouching.txt", "Actor9Idle.txt", "Actor9Running.txt", "Actor9Swing.txt", "Actor9Walking.txt"]
base_url = "https://raw.githubusercontent.com/RafaelDrumond/PeekDB/master/Motion/"

classes = ["Crouching", "Idle", "Running", "Siwing", "Walking"]

actions = classes
window  = 40

def build_url(actor, action):
    assert action in classes
    assert actor in range(11)
    return base_url+"Actor"+str(actor)+action+".txt"


actors = range(3,11)
n_actions = len(actions)
xt = np.empty((len(actors)*n_actions,20,window))
yt = np.empty((len(actors)*n_actions,n_actions))
for action, action_name in enumerate(actions):
    class_vector         = [0] * len(actions)
    class_vector[action] = 1
    yt[action, :]        = class_vector
    for actor, actor_n in enumerate(actors):
        url = build_url(actor_n, action_name)
        if url in [base_url+f for f in files]:
            response = urllib.request.urlopen(url)
            response = response.read().decode("utf8")
            response = response.split("\r\n")
            for i,line in enumerate(response[0:window]):
                line = line.split("|m ")[1].split(" |class")[0].split(" ")
                xt[actor*n_actions+action, :, i] = line

actors = [2]
n_actions = len(actions)
xv = np.empty((len(actors)*n_actions, 20, window)) # event, features, time
yv = np.empty((len(actors)*n_actions, n_actions))
for action, action_name in enumerate(actions):
    class_vector         = [0] * len(actions)
    class_vector[action] = 1
    yv[action, :]        = class_vector
    for actor, actor_n in enumerate(actors):
        url = build_url(actor_n, action_name)
        if url in [base_url+f for f in files]:
            response = urllib.request.urlopen(url)
            response = response.read().decode("utf8")
            response = response.split("\r\n")
            for i,line in enumerate(response[0:window]):
                line = line.split("|m ")[1].split(" |class")[0].split(" ")
                xv[actor*n_actions+action, :, i] = line


x = tf.placeholder(tf.float32, [None, 20, window], name="Inputs")
y = tf.placeholder(tf.float32, [None, n_actions], name="Labels")

model = tf.keras.layers.LSTM(200, name="LSTM1")(x)
model = tf.layers.Dropout(.5, name="dropout1")(model)
model = tf.layers.Dense(27, name="dense1")(model)
model = tf.layers.Dense(n_actions, name="dense2")(model)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=model), name="loss")

# optimizer
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Accuracy
prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name="accuracy")


epochs = 50
print("Begin Session")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    counter = 0
    tic = time()
    for epoch in range(epochs):
        print("Start Epoch {:>2}".format(epoch+1))
        t = time()
        sess.run(optimizer,feed_dict={x: xt, y: yt})
        vloss, vacc = sess.run([loss, accuracy], feed_dict={x: xv, y: yv})
        tloss, tacc = sess.run([loss, accuracy], feed_dict={x: xt, y: yt})
        print("Validation Loss: {:>8.4f}, Validation Accuracy: {:>8.6f}".format(vloss, vacc))
        print("Training   Loss: {:>8.4f}, Training   Accuracy: {:>8.6f}".format(tloss, tacc))
        print("Epoch time: ", time() - t)
    print("Training time:", time() - tic)
