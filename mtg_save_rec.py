# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import tensorflow as tf
import os
import word_helpers
import pickle


# PATHS -- absolute
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path,"saved","mtg","mtg_rec.ckpt")
data_path = os.path.join(dir_path,"data","cards_tokenized.txt")

# Load mtg tokenized data
# Special thanks to mtgencode: https://github.com/billzorn/mtgencode
with open(data_path,"r") as f:
    # Each card occupies its own line in this tokenized version
    raw_txt = f.read()#.split("\n")

try:
    with open(os.path.join(dir_path,"data","mtg_tokenized_wh.pkl"),"rb") as f:
        WH = pickle.load(f)
except Exception as e:
    print(e)
    # What's with those weird symbols?
    # u'\xbb' is our GO symbol (»)
    # u'\xac' is our UNKNOWN symbol (¬)
    # u'\xa4' is our END symbol (¤)
    # They're arbitrarily chosen, but
    # I think they both:
    #   1). Are unlikely to appear in regular data, let alone cleaned data.
    #   2). Look awesome.
    vocab = [u'\xbb','|', '5', 'c', 'r', 'e', 'a', 't', 'u', '4', '6', 'h', 'm', 'n', ' ', 'o', 'd', 'l', 'i', '7', \
             '8', '&', '^', '/', '9', '{', 'W', '}', ',', 'T', ':', 's', 'y', 'b', 'f', 'v', 'p', '.', '3', \
             '0', 'A', '1', 'w', 'g', '\\', 'E', '@', '+', 'R', 'C', 'x', 'B', 'G', 'O', 'k', '"', 'N', 'U', \
             "'", 'q', 'z', '-', 'Y', 'X', '*', '%', '[', '=', ']', '~', 'j', 'Q', 'L', 'S', 'P', '2',u'\xac',u'\xa4']

    WH = word_helpers.WordHelper(raw_txt, vocab)
    # Save our WordHelper
    with open(os.path.join(dir_path,"data","mtg_tokenized_wh.pkl"),"wb") as f:
        pickle.dump(WH,f)


# Network Parameters
LEARNING_RATE = 0.001
N_INPUT = WH.vocab.vocab_size # One-hot encoded letter
N_CLASSES = WH.vocab.vocab_size # Number of possible characters
LSTM_SIZE = 256
NUM_LAYERS = 3
BATCH_SIZE = 1
NUM_EPOCHS = 1000
MINI_BATCH_LEN = 100
DISPLAY_STEP = 10
MAX_LENGTH = 150

graph = tf.Graph()
with graph.as_default():
    # tf Graph input
    x = tf.placeholder(tf.float32, [None,  N_INPUT])
    y = tf.placeholder(tf.float32, [None, N_CLASSES])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([LSTM_SIZE,  N_CLASSES]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([N_CLASSES]))
    }

    def model(x, weights, biases):
        # Cell definition
        lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE, state_is_tuple=True)

        # Full model definition
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * NUM_LAYERS, state_is_tuple=True)

        outputs, states = tf.contrib.rnn.static_rnn(stacked_lstm, [x], dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    out = model(x, weights, biases)

    # Create the softmax vector for prediction comparison
    pred = tf.nn.softmax(logits=out)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()

#Running first session
with tf.Session(graph=graph) as sess:
    # Initialize variables
    sess.run(init)

    try:
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)
    except Exception as e:
        print("Model restore failed {}".format(e))

    # Training cycle
    for epoch in range(NUM_EPOCHS):
        avg_cost = 0
        for b in range(MINI_BATCH_LEN):
            batch_x, batch_y = WH.TrainBatches.next()
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c
        avg_cost /= MINI_BATCH_LEN
        # Display logs per epoch step
        if epoch % DISPLAY_STEP == 0:
            print(" ") # Spacer
            print("Epoch:", '%04d' % (epoch+1), "cost=" , "{:.9f}".format(avg_cost))
            # Test model
            preds = []
            true = []
            avg_acc = 0
            for b in range(MINI_BATCH_LEN):
                batch_x, batch_y = WH.TestBatches.next()
                p,_,acc = sess.run([pred,correct_prediction,accuracy], feed_dict={x: batch_x, y: batch_y})
                preds.append(np.random.choice(WH.vocab.vocab, 1, p=p[0])[0])
                true.append(WH.vocab.onehot2char(batch_y))
                avg_acc += acc
            print("PRED: {}".format(''.join(preds)))
            print("TRUE: {}".format(''.join(true)))
            print("Accuracy:", acc/MINI_BATCH_LEN)

    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
