# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import tensorflow as tf
import os
import word_helpers
import pickle
from rec_char import weighted_pick,SAVE_DIR,CHECKPOINT_NAME,DATA_NAME,PICKLE_PATH,SUBDIR_NAME
from rec_char import NUM_LAYERS, LSTM_SIZE
print("Beginning Session")

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path,"saved",SAVE_DIR,CHECKPOINT_NAME)
checkpoint_path = os.path.join(dir_path,"saved",SAVE_DIR)
data_path = os.path.join(dir_path,"data",SUBDIR_NAME,DATA_NAME)
try:
    with open(os.path.join(checkpoint_path,PICKLE_PATH),"rb") as f:
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
    vocab = "1 2 3 4 5 6 7 8 9 0".split(" ")
    vocab += "a b c d e f g h i j k l m n o p q r s t u v w x y z".split(" ")
    vocab += "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split(" ")
    vocab += ['|', ' ', '&', '^', '/', '{', '}', ',', ':', '.', '\\',  '@', '+', '"', "'", '-', '*', '%', '[', '=', ']', '~']
    vocab += [u'\xbb',  u'\xac', u'\xf8', u'\xa4', u'\u00BB']
    # Load mtg tokenized data
    # Special thanks to mtgencode: https://github.com/billzorn/mtgencode
    with open(data_path,"r") as f:
         # Each card occupies its own line in this tokenized version
         raw_txt = f.read()#.split("\n")
    WH = word_helpers.WordHelper(raw_txt, vocab)
    #WH = word_helpers.JSONHelper(data_path,vocab)



dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path,"saved",SAVE_DIR,CHECKPOINT_NAME)
checkpoint_path = os.path.join(dir_path,"saved",SAVE_DIR)
data_path = os.path.join(dir_path,"data",SUBDIR_NAME,DATA_NAME)

#PRIME_TEXT = "»|5creature|4legendary|6eldrazi|7|8"
#PRIME_TEXT = u"»|5planeswalker|4|6"
PRIME_TEXT = "»|5creature|4|6"
#PRIME_TEXT = u"»|5planeswalker|4|6serra|7"

TEMPERATURE = 0.5
NUM_PRED = 200

vocab = WH.vocab.vocab
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

lr = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model/learning_rate")[0]

final_state = graph.get_tensor_by_name('model/final_state:0')
pred = graph.get_tensor_by_name('model/pred:0')
x = graph.get_tensor_by_name('model/input_placeholder:0')
init_state = graph.get_tensor_by_name('model/state_placeholder:0')
dropout_prob = graph.get_tensor_by_name('model/dropout:0')
temp = graph.get_tensor_by_name('model/temp:0')

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
    state = np.zeros((NUM_LAYERS,2,1,LSTM_SIZE))

    # Begin our primed text Feeding
    for c in PRIME_TEXT[:-1]:
        prime_x = np.array([vocab.index(c)]).reshape((1,1))
        s, = sess.run([final_state], feed_dict={x: prime_x,
                                                   init_state: state,
                                                   dropout_prob: 1.0,
                                                   temp:TEMPERATURE})
        state = s

    # We iterate over every pair of letters in our test batch
    init_x = np.array([vocab.index(PRIME_TEXT[-1])]).reshape((1,1))
    for i in range(0,NUM_PRED):
        s,p = sess.run([final_state, pred], feed_dict={x: init_x,
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
