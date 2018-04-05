# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import tensorflow as tf
import os
from helpers import word_helpers, rap_parser
import pickle
import time
import caffeine

# PATHS -- absolute
# SAVE_DIR = "dict" # mtg
# CHECKPOINT_NAME = "dict_steps.ckpt" # "mtg_rec_char_steps.ckpt"
# DATA_NAME = "dictionary.json" # "cards_tokenized.txt"
# PICKLE_PATH = "dict_tokenized.pkl" #"mtg_tokenized_wh.pkl"
# SUBDIR_NAME = "dictionary"
#
# dir_path = os.path.dirname(os.path.realpath(__file__))
# model_path = os.path.join(dir_path,"saved",SAVE_DIR,CHECKPOINT_NAME)
# checkpoint_path = os.path.join(dir_path,"saved",SAVE_DIR)
# data_path = os.path.join(dir_path,"data",SUBDIR_NAME,DATA_NAME)

SAVE_DIR = "rap"
CHECKPOINT_NAME = "rap_char_steps.ckpt"
DATA_NAME = "ohhla.txt"
PICKLE_PATH = "rap_wh.pkl"
SUBDIR_NAME = "rap"

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path,"saved",SAVE_DIR,CHECKPOINT_NAME)
checkpoint_path = os.path.join(dir_path,"saved",SAVE_DIR)
data_path = os.path.join(dir_path,"data",SUBDIR_NAME,DATA_NAME)



try:
    with open(os.path.join(checkpoint_path,PICKLE_PATH),"rb") as f:
        WH = pickle.load(f)
except Exception as e:
    print(e)
    songs,parsed_vocab = rap_parser.getSongs()
    raw_txt = "\n".join(songs)
    print("parsed vocab length: {}".format(len(parsed_vocab)))
    WH = word_helpers.WordHelper(raw_txt,vocab_list=parsed_vocab)
    #WH = word_helpers.JSONHelper(data_path,vocab)

    # Save our WordHelper
    with open(os.path.join(checkpoint_path,PICKLE_PATH),"wb") as f:
        pickle.dump(WH,f)


# Network Parameters
LEARNING_RATE = 3e-4
GRAD_CLIP = 5.0
LSTM_SIZE = 512
NUM_LAYERS = 3
BATCH_SIZE = 100 # Feeding a single character across multiple batches at a time
NUM_EPOCHS = 100
NUM_STEPS = 100 # 50
DISPLAY_STEP = 10#25
SAVE_STEP = 1
DECAY_RATE = 0.97
DECAY_STEP = 5
DROPOUT_KEEP_PROB = 0.8 #0.5
TEMPERATURE = 1.0
NUM_PRED = 50
already_trained = 0
PRIME_TEXT = u"»"
N_CLASSES = 98
vocab = WH.vocab.vocab

def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1)*s)))

with tf.variable_scope("model") as super_scope:
    # Placeholders
    x = tf.placeholder(tf.int32, [None, 1], name='input_placeholder')
    y = tf.placeholder(tf.int32, [None, 1], name='labels_placeholder')
    dropout_prob = tf.placeholder(tf.float32, name="dropout")

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
    #onehot = tf.reshape(tf.one_hot(x, N_CLASSES),[-1,N_CLASSES])
    #rnn_inputs  = tf.reshape(tf.layers.dense(inputs=onehot, units=LSTM_SIZE,activation=tf.nn.sigmoid),[-1,1,LSTM_SIZE])
    embedding = tf.get_variable("embedding", [N_CLASSES, LSTM_SIZE])
    rnn_inputs = tf.nn.embedding_lookup(embedding, x)

    #embedding = tf.get_variable("embedding", [N_CLASSES, LSTM_SIZE])
    #rnn_inputs = tf.nn.embedding_lookup(embedding, x)

    # RNN
    lstm = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
    dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_prob)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([dropout] * NUM_LAYERS)

    state = rnn_tuple_state
    output_list = []
    output, state = stacked_lstm(rnn_inputs[:,0], state)
    output_list.append(tf.reshape(output,[-1,1,LSTM_SIZE]))
    final_state = tf.identity(state, name="final_state")

    rnn_outputs = tf.concat(output_list,axis=1)

    # rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=stacked_lstm,
    #                                              inputs=rnn_inputs,
    #                                              initial_state=rnn_tuple_state)#stacked_lstm.zero_state(batch_size,tf.float32))
    #

    temp = tf.placeholder(tf.float32, name="temp")

    #Predictions, loss, training step
    with tf.variable_scope("dense") as scope:
        flat = tf.reshape(rnn_outputs, [-1, LSTM_SIZE])
        dense = tf.layers.dense(inputs=flat, units=N_CLASSES)
        logits = tf.reshape(dense,[-1,batch_size, N_CLASSES])
    #pred = tf.nn.softmax(logits)
    pred = tf.nn.softmax(tf.div(logits,temp),name="pred")

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    cost = tf.div(tf.reduce_sum(losses ), tf.cast(batch_size,tf.float32))

    lr = tf.Variable(0.0, name="learning_rate", trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), GRAD_CLIP)
    with tf.name_scope('optimizer'):
        op = tf.train.AdamOptimizer(lr)
    optimizer = op.apply_gradients(zip(grads, tvars))

    # Initialize variables
    #init = tf.global_variables_initializer()
    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()



if __name__ == "__main__":
    print("Beginning Session")
    #  TRAINING Parameters
    #vocab,batches = getTrainingData()
    NUM_BATCHES = WH.TrainBatches.num_batches // BATCH_SIZE
    # N_CLASSES = len(vocab)

    #Running first session
    with tf.Session() as sess:
        # Initialize variables
        #sess.run(init)
        sess.run(tf.global_variables_initializer())

        try:
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            # A quirk of training on a different machine:
            # the model_checkpoint_path is an absolute path and
            # makes restoring fail because it doesn't match the path here.
            # To avoid this, we extract the checkpoint file name
            # then recreate the correct path and restore from there.
            restore_path = ckpt.model_checkpoint_path
            restore_file = os.path.basename(restore_path)
            ckpt_file = os.path.basename(restore_path)
            already_trained = int(ckpt_file.replace(CHECKPOINT_NAME+"-",""))
            new_path = os.path.join(dir_path,"saved",SAVE_DIR,restore_file)
            saver.restore(sess, new_path)#ckpt.model_checkpoint_path)
            print("Model restored from file: %s" % model_path)
            print("__________________________________________")
        except Exception as e:
            print("Model restore failed {}".format(e))
            print("__________________________________________")

        # Training cycle
        for epoch in range(already_trained,already_trained+NUM_EPOCHS):
            # Set learning rate
            sess.run(tf.assign(lr,LEARNING_RATE * (DECAY_RATE ** (epoch % DECAY_STEP))  ))

            start = time.time()
            sum_cost = 0
            count = 0

            for batch_ind in range(NUM_BATCHES): # Get a batch
                batch = WH.TrainBatches.next_card_batch(BATCH_SIZE,NUM_STEPS)
                # Reset state value
                state = np.zeros((NUM_LAYERS,2,len(batch),LSTM_SIZE))
                batch_cost = 0
                seq_len = batch.shape[1]
                for i in range(0,seq_len-1):
                    count += 1
                    # Simulate truncated backprop through time by
                    # reseting the state after NUM_STEPS
                    if count != 0 and count % NUM_STEPS == 0:
                        state = np.zeros((NUM_LAYERS,2,len(batch),LSTM_SIZE)) # reset state
                    batch_x = batch[:,i].reshape((BATCH_SIZE,1))
                    batch_y = batch[:,(i+1)].reshape((BATCH_SIZE,1))
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, s, c = sess.run([optimizer, final_state, cost], feed_dict={x: batch_x,
                                                                                  y: batch_y,
                                                                                  init_state: state,
                                                                                  dropout_prob: DROPOUT_KEEP_PROB,
                                                                                  temp:1.0})
                    state = s
                    sum_cost += c
                    batch_cost += c
                print("     batch {} of {} processed, cost: {}, epoch {}".format(batch_ind,NUM_BATCHES,batch_cost/(seq_len-1), epoch))
                # Display logs per epoch step
                if batch_ind % DISPLAY_STEP == 0:
                    # Test model
                    preds = []
                    true = []

                    # We no longer use BATCH_SIZE here because
                    # in the test method we only want to compare
                    # one card output to one card prediction
                    preds = [c for c in PRIME_TEXT]
                    #unused_y = np.zeros((1,1))
                    state = np.zeros((NUM_LAYERS,2,1,LSTM_SIZE))

                    # Begin our primed text Feeding
                    for c in PRIME_TEXT[:-1]:
                        prime_x = np.array([vocab.index(c)]).reshape((1,1))
                        s, = sess.run([final_state], feed_dict={x: prime_x,
                                                                   #y: unused_y,
                                                                   init_state: state,
                                                                   dropout_prob: 1.0,
                                                                   temp:TEMPERATURE})
                        state = s

                    # We iterate over every pair of letters in our test batch
                    init_x = np.array(vocab.index(PRIME_TEXT[-1])).reshape((1,1))
                    for i in range(0,NUM_PRED):
                        s,l,p = sess.run([final_state,logits, pred], feed_dict={x: init_x,
                                                                       #y: unused_y,
                                                                       init_state: state,
                                                                       dropout_prob: 1.0,
                                                                       temp:TEMPERATURE})
                        state = s
                        # Choose a letter from our vocabulary based on our output probability: p
                        for j in p:
                            pred_index = weighted_pick(j)
                            pred_letter = vocab[pred_index]
                            preds.append(pred_letter)
                            init_x = np.array([[pred_index]])
                    print(" ") # Spacer
                    print("PRED: {}".format(''.join(preds)))
                    save_path = saver.save(sess, model_path, global_step = epoch)
                    #print("TRUE: {}".format(''.join(true)))

            end = time.time()
            avg_cost = (sum_cost/BATCH_SIZE)/NUM_BATCHES
            print("Epoch:", '%04d' % (epoch), "cost=" , "{:.9f}".format(avg_cost), "time:", "{}".format(end-start))

            if epoch % SAVE_STEP == 0 and epoch != 0:
                save_path = saver.save(sess, model_path, global_step = epoch)

        # Save model weights to disk
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
