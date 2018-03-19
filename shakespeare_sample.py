# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import tensorflow as tf
import os
from helpers import word_helpers
import pickle
import time
#import caffeine

from shakespeare import weighted_pick,getTrainingData, SAVE_DIR,CHECKPOINT_NAME,DATA_NAME,PICKLE_PATH,SUBDIR_NAME,VOCAB_NAME,BATCHES_NAME
from shakespeare import NUM_LAYERS, LSTM_SIZE
print("Beginning Session")
#  TRAINING Parameters

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path,"saved",SAVE_DIR,CHECKPOINT_NAME)
checkpoint_path = os.path.join(dir_path,"saved",SAVE_DIR)
data_path = os.path.join(dir_path,"data",SUBDIR_NAME,DATA_NAME)
PRIME_TEXT = "SCENE I."
TEMPERATURE = 1.0
NUM_PRED = 1000

vocab,batches = getTrainingData()
NUM_BATCHES = len(batches)
N_CLASSES = len(vocab)

ckpt = tf.train.get_checkpoint_state(checkpoint_path)
restore_path = ckpt.model_checkpoint_path
restore_file = os.path.basename(restore_path)
ckpt_file = os.path.basename(restore_path)
already_trained = int(ckpt_file.replace(CHECKPOINT_NAME+"-",""))
print("EPOCHS TRAINED: {}".format(already_trained))
saver = tf.train.import_meta_graph(os.path.join(checkpoint_path,"{}-{}.meta".format(CHECKPOINT_NAME,already_trained)))

# We can now access the default graph where all our metadata has been loaded
graph = tf.get_default_graph()

lr = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="learning_rate")[0]

final_state = graph.get_tensor_by_name('final_state:0')
pred = graph.get_tensor_by_name('pred:0')
x = graph.get_tensor_by_name('input_placeholder:0')
y = graph.get_tensor_by_name('labels_placeholder:0')
init_state = graph.get_tensor_by_name('state_placeholder:0')
dropout_prob = graph.get_tensor_by_name('dropout:0')
temp = graph.get_tensor_by_name('temp:0')

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


    # Set learning rate
    sess.run(tf.assign(lr,0))

    # Test model
    preds = []
    true = []

    # We no longer use BATCH_SIZE here because
    # in the test method we only want to compare
    # one card output to one card prediction
    preds = [c for c in PRIME_TEXT]
    unused_y = np.zeros((1,1))
    state = np.zeros((NUM_LAYERS,2,1,LSTM_SIZE))

    # Begin our primed text Feeding
    for c in PRIME_TEXT[:-1]:
        prime_x = np.array([vocab.index(c)]).reshape((1,1))
        s, = sess.run([final_state], feed_dict={x: prime_x,
                                                   y: unused_y,
                                                   init_state: state,
                                                   dropout_prob: 1.0,
                                                   temp:TEMPERATURE})
        state = s

    # We iterate over every pair of letters in our test batch
    init_x = np.array([vocab.index(PRIME_TEXT[-1])]).reshape((1,1))
    for i in range(0,NUM_PRED):
        s,p = sess.run([final_state, pred], feed_dict={x: init_x,
                                                       y: unused_y,
                                                       init_state: state,
                                                       dropout_prob: 1.0,
                                                       temp:TEMPERATURE})

        # Choose a letter from our vocabulary based on our output probability: p
        for j in p:
            #pred_index = weighted_pick(j)
            pred_index = np.random.choice(len(vocab),1, p=j[0])[0]
            pred_letter = vocab[pred_index]
            preds.append(pred_letter)
            init_x = np.array([[pred_index]])
        state = s

    print(" ") # Spacer
    print("PRED: {}".format(''.join(preds)))
    #print("TRUE: {}".format(''.join(true)))
