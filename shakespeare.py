# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import tensorflow as tf
import os
import word_helpers
import pickle
import time
#import caffeine


SAVE_DIR = "shakespeare"
CHECKPOINT_NAME = "shakespeare_rec_char_steps.ckpt"
DATA_NAME = "shakey_bill.txt"
PICKLE_PATH = "shakespeare_tokenized_wh.pkl"
SUBDIR_NAME = "shakespeare"

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path,"saved",SAVE_DIR,CHECKPOINT_NAME)
checkpoint_path = os.path.join(dir_path,"saved",SAVE_DIR)
data_path = os.path.join(dir_path,"data",SUBDIR_NAME,DATA_NAME)


try:
    with open(os.path.join(dir_path,"data",PICKLE_PATH),"rb") as f:
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
             "'", 'q', 'z', '-', 'Y', 'X', '*', '%', '[', '=', ']', '~', 'j', 'Q', 'L', 'S', 'P', '2',u'\xac',u'\xf8',u'\xa4',u'\u00BB']

    # Load mtg tokenized data
    # Special thanks to mtgencode: https://github.com/billzorn/mtgencode
    with open(data_path,"r") as f:
         # Each card occupies its own line in this tokenized version
         raw_txt = f.read()#.split("\n")
    WH = word_helpers.WordHelper(raw_txt, vocab)
    #WH = word_helpers.JSONHelper(data_path,vocab)


    # Save our WordHelper
    with open(os.path.join(dir_path,"data",PICKLE_PATH),"wb") as f:
        pickle.dump(WH,f)

# max_len = 75 # Determined by going through entire corpus
# batch = WH.TrainBatches.next_card_batch(100,1)
# for b in batch:
#     print("".join([WH.vocab.vocab[i] for i in b]))
# HODOR

def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1)*s)))

args = {
    'learning_rate':1e-3,#3e-4
    'grad_clip':5.0,
    'n_input':WH.vocab.vocab_size,
    'n_classes':WH.vocab.vocab_size,
    'lstm_size':512, #128,
    'num_layers':3, #2
    'num_steps':50 #250
}


# Network Parameters
LEARNING_RATE = args['learning_rate']
GRAD_CLIP = args['grad_clip']
N_INPUT = args['n_input']
N_CLASSES = args['n_classes']
LSTM_SIZE = args['lstm_size']
NUM_LAYERS = args['num_layers']
NUM_STEPS = args['num_steps']
BATCH_SIZE = 100 # Feeding a single character across multiple batches at a time
NUM_EPOCHS = 100
DISPLAY_STEP = 10#25
SAVE_STEP = 1
DECAY_RATE = 0.97
DECAY_STEP = 5
DROPOUT_KEEP_PROB = 1.0 #0.5
TEMPERATURE = 1.0
NUM_PRED = 50
already_trained = 0
PRIME_TEXT = "»To be or not to be"

with tf.device('/cpu:0'):
    graph = tf.Graph()
    with graph.as_default():
        # Placeholders
        x = tf.placeholder(tf.int32, [None, 1], name='input_placeholder')
        y = tf.placeholder(tf.int32, [None, 1], name='labels_placeholder')
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
        final_state = state
        rnn_outputs = tf.concat(output_list,axis=1)

        # rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=stacked_lstm,
        #                                              inputs=rnn_inputs,
        #                                              initial_state=rnn_tuple_state)#stacked_lstm.zero_state(batch_size,tf.float32))
        #

        temp = tf.placeholder(tf.float32)

        #Predictions, loss, training step
        with tf.variable_scope("dense") as scope:
            flat = tf.reshape(rnn_outputs, [-1, LSTM_SIZE])
            dense = tf.layers.dense(inputs=flat, units=N_CLASSES)
            logits = tf.reshape(dense,[-1,batch_size, N_CLASSES])
        #pred = tf.nn.softmax(logits)
        pred = tf.nn.softmax(tf.div(logits,temp))

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        cost = tf.div(tf.reduce_sum(losses ), tf.cast(batch_size,tf.float32))

        lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), GRAD_CLIP)
        with tf.name_scope('optimizer'):
            op = tf.train.AdamOptimizer(lr)
        optimizer = op.apply_gradients(zip(grads, tvars))

        # Initialize variables
        init = tf.global_variables_initializer()
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()



    print("Beginning Session")
    #  TRAINING Parameters

    #Running first session
    with tf.Session(graph=graph) as sess:
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
            sess.run(tf.assign(lr,LEARNING_RATE * (DECAY_RATE ** (epoch % DECAY_STEP)   )))

            start = time.time()
            sum_cost = 0
            num_batches = WH.TrainBatches.num_batches // BATCH_SIZE
            for _batch in range(num_batches):
                # Generate a batch

                batch = WH.TrainBatches.next_card_batch(BATCH_SIZE,NUM_STEPS)
                # Reset state value
                state = np.zeros((NUM_LAYERS,2,len(batch),LSTM_SIZE))
                batch_cost = 0
                for i in range(0,batch.shape[1]-1):
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
                print("     batch {} of {} processed, cost: {}, epoch {}".format(_batch,num_batches,batch_cost/(batch.shape[1]-1), epoch))
                # Display logs per epoch step
                if _batch % DISPLAY_STEP == 0:
                    # Test model
                    preds = []
                    true = []

                    # We no longer use BATCH_SIZE here because
                    # in the test method we only want to compare
                    # one card output to one card prediction
                    init_x = np.array([WH.TrainBatches.vocab.go]).reshape((1,1))
                    preds = [c for c in PRIME_TEXT] + [" "]
                    unused_y = np.zeros((1,1))
                    state = np.zeros((NUM_LAYERS,2,1,LSTM_SIZE))

                    # Begin our primed text Feeding
                    for c in PRIME_TEXT:
                        prime_x = np.array([WH.vocab.vocab.index(c)]).reshape((1,1))
                        s, = sess.run([final_state], feed_dict={x: prime_x,
                                                                   y: unused_y,
                                                                   init_state: state,
                                                                   dropout_prob: 1.0,
                                                                   temp:TEMPERATURE})
                        state = s

                    # We iterate over every pair of letters in our test batch
                    init_x = np.array([WH.TrainBatches.vocab.vocab.index(' ')]).reshape((1,1))
                    for i in range(0,NUM_PRED):
                        s,l,p = sess.run([final_state,logits, pred], feed_dict={x: init_x,
                                                                       y: unused_y,
                                                                       init_state: state,
                                                                       dropout_prob: 1.0,
                                                                       temp:TEMPERATURE})
                        state = s
                        # Choose a letter from our vocabulary based on our output probability: p
                        for j in p:
                            #pred_letter = np.random.choice(WH.vocab.vocab, 1, p=j[0])[0]
                            pred_index = weighted_pick(j)
                            pred_letter = WH.vocab.vocab[pred_index]
                            #pred_letter = WH.vocab.vocab[np.argmax(j[0])]
                            preds.append(pred_letter)
                            init_x = np.array([[pred_index]])
                        #for l in range(batch_y.shape[1]):
                        #    true.append(WH.vocab.id2char(batch_y[0][l]))
                    print(" ") # Spacer
                    print("PRED: {}".format(''.join(preds)))
                    #print("TRUE: {}".format(''.join(true)))

            end = time.time()
            avg_cost = (sum_cost/BATCH_SIZE)/num_batches
            print("Epoch:", '%04d' % (epoch), "cost=" , "{:.9f}".format(avg_cost), "time:", "{}".format(end-start))

            if epoch % SAVE_STEP == 0 and epoch != 0:
                save_path = saver.save(sess, model_path, global_step = epoch)

        # Save model weights to disk
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
