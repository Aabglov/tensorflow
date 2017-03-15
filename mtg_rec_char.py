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
model_path = os.path.join(dir_path,"saved","mtg","mtg_rec_char.ckpt")
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
             "'", 'q', 'z', '-', 'Y', 'X', '*', '%', '[', '=', ']', '~', 'j', 'Q', 'L', 'S', 'P', '2',u'\xac',u'\xf8',u'\xa4']

    WH = word_helpers.WordHelper(raw_txt, vocab)
    # Save our WordHelper
    with open(os.path.join(dir_path,"data","mtg_tokenized_wh.pkl"),"wb") as f:
        pickle.dump(WH,f)


# Network Parameters
LEARNING_RATE = 0.001
N_INPUT = WH.vocab.vocab_size # One-hot encoded letter
N_CLASSES = WH.vocab.vocab_size # Number of possible characters
LSTM_SIZE = 512
NUM_LAYERS = 3
NUM_STEPS = 1 # Only feeding in one character at a time
BATCH_SIZE = 12 # Feeding a single character across 12 batches at a time
NUM_EPOCHS = 100000
MINI_BATCH_LEN = 100
DISPLAY_STEP = 100
MAX_LENGTH = 150

graph = tf.Graph()
with graph.as_default():
    # Placeholders
    x = tf.placeholder(tf.int32, [None, NUM_STEPS], name='input_placeholder')
    y = tf.placeholder(tf.int32, [None, ], name='labels_placeholder')
    dropout_prob = tf.placeholder(tf.float32)

    # Our initial state placeholder:
    # NUM_LAYERS -- the number of layers used by our stacked LSTM
    # 2 -- 2 states for each layer (output,hidden)
    # None -- This will be our batch_size which is flexible
    # LSTM_SIZE -- the size of our hidden layers
    init_state = tf.placeholder(tf.float32, [NUM_LAYERS, 2, None, LSTM_SIZE], name='state_placeholder')
    # Create appropriate LSTMStateTuple for dynamic_rnn function out of our placeholder
    l = tf.unstack(init_state, axis=0)
    rnn_tuple_state = tuple(
        [tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1]) for idx in range(NUM_LAYERS)]
    )

    # Get dynamic batch_size
    batch_size = tf.shape(x)[0]



    #Inputs
    rnn_inputs = tf.one_hot(x, N_CLASSES)

    # RNN
    lstm = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
    dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_prob)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([dropout] * NUM_LAYERS)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=stacked_lstm,
                                                 inputs=rnn_inputs,
                                                 initial_state=rnn_tuple_state)#stacked_lstm.zero_state(batch_size,tf.float32))

    #Predictions, loss, training step
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [NUM_STEPS*LSTM_SIZE, N_CLASSES])
        b = tf.get_variable('b', [N_CLASSES], initializer=tf.constant_initializer(0.0))
    logits = tf.reshape(
                tf.matmul(tf.reshape(rnn_outputs, [-1, NUM_STEPS*LSTM_SIZE]), W) + b,
                [batch_size, N_CLASSES])
    pred = tf.nn.softmax(logits)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    cost = tf.reduce_mean(losses)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    # Initialize variables
    init = tf.global_variables_initializer()
    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()


print("Beginning Session")
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
        # Generate a batch
        batch = WH.TrainBatches.next_card_batch(BATCH_SIZE)
        # Reset state value
        state = np.zeros((NUM_LAYERS,2,len(batch),LSTM_SIZE))
        for i in range(len(batch)-1): # -1 because the y column will come from the 'next' element
            #print("BATCH_SIZE: {}, Batch shape: {}".format(BATCH_SIZE,batch.shape))
            batch_x = batch[:,i].reshape((BATCH_SIZE,1))
            batch_y = batch[:,i+1]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, s, c = sess.run([optimizer, final_state, cost], feed_dict={x: batch_x, y: batch_y, init_state: state, dropout_prob: 0.5})
            state = s
        avg_cost = c/BATCH_SIZE
        # Display logs per epoch step
        if epoch % DISPLAY_STEP == 0:
            print(" ") # Spacer
            print("Epoch:", '%04d' % (epoch+1), "cost=" , "{:.9f}".format(avg_cost))
            # Test model
            preds = []
            true = []


            # We no longer use BATCH_SIZE here because
            # in the test method we only want to compare
            # one card output to one card prediction
            test_batch = WH.TestBatches.next_card_batch(1)
            state = np.zeros((NUM_LAYERS,2,1,LSTM_SIZE))
            # We iterate over every pair of letters in our test batch
            for i in range(test_batch.shape[1]-1): # -1 because the y column will come from the 'next' element
                batch_x = test_batch[0,i].reshape((1,1)) # Reshape to (?,batch_size), in this case (?,1)
                batch_y = test_batch[0,i+1].reshape((1,)) # Reshape to (?,), in this case (1,)
                s,p = sess.run([final_state, pred], feed_dict={x: batch_x, y: batch_y, init_state: state, dropout_prob: 1.0})
                state = s
                # Choose a letter from our vocabulary based on our output probability: p
                pred_letter = np.random.choice(WH.vocab.vocab, 1, p=p[0])[0]
                preds.append(pred_letter)
            for l in test_batch[0][1:]: # Test batch is 2D so the 0 gets the first (only) line and the first charater is never preidcted (GO) so we omit it as well
                true.append(WH.vocab.id2char(l))
            print("PRED: {}".format(''.join(preds)))
            print("TRUE: {}".format(''.join(true)))

            seed = u"»"#"|5creature"
            start = np.array([WH.vocab.go]).reshape((1,1)) # 0 is our go character
            state = np.zeros((NUM_LAYERS,2,1,LSTM_SIZE))
            sample = []
            for _ in range(100):
                s, p = sess.run([final_state, pred], feed_dict={x: start, init_state: state, dropout_prob: 1.0})
                pred_letter = np.random.choice(WH.vocab.vocab, 1, p=p[0])[0]
                pred_id = WH.vocab.char2id(pred_letter)
                start = np.array([pred_id]).reshape((1,1))
                sample.append(pred_letter)
            print("SAMPLE: {}".format(seed + ''.join(sample)))
            save_path = saver.save(sess, model_path)

    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
