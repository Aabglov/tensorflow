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
checkpoint_path = os.path.join(dir_path,"saved","mtg")#,"mtg_rec_char_steps.ckpt")
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


args = {
    'learning_rate':0.000,#0.001
    'decay_rate':1.0,
    'grad_clip':0.0,
    'n_input':WH.vocab.vocab_size,
    'n_classes':WH.vocab.vocab_size,
    'lstm_size':256,
    'num_layers':3,
    'num_steps':1
}
# Network Parameters
LEARNING_RATE = args['learning_rate']
DECAY_RATE = args['decay_rate']
GRAD_CLIP = args['grad_clip']
N_INPUT = args['n_input']
N_CLASSES = args['n_classes']
LSTM_SIZE = args['lstm_size']
NUM_LAYERS = args['num_layers']
NUM_STEPS = args['num_steps']

graph = tf.Graph()
with graph.as_default():
    # Placeholders
    x = tf.placeholder(tf.int32, [None, NUM_STEPS], name='input_placeholder')
    y = tf.placeholder(tf.int32, [None, NUM_STEPS], name='labels_placeholder')
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
    embedding = tf.get_variable("embedding", [N_CLASSES, LSTM_SIZE])
    rnn_inputs = tf.nn.embedding_lookup(embedding, x)

    # RNN
    lstm = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
    dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_prob)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([dropout] * NUM_LAYERS)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=stacked_lstm,
                                                 inputs=rnn_inputs,
                                                 initial_state=rnn_tuple_state)#stacked_lstm.zero_state(batch_size,tf.float32))

    #Predictions, loss, training step
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [LSTM_SIZE, N_CLASSES])
        b = tf.get_variable('b', [N_CLASSES], initializer=tf.constant_initializer(0.0))
    logits = tf.reshape(
                tf.matmul(tf.reshape(rnn_outputs, [-1, LSTM_SIZE]), W) + b,
                [-1,batch_size, N_CLASSES])
    pred = tf.nn.softmax(logits)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    cost = tf.reduce_mean(losses)

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


print("Beginning Model Initialization")
#Running first session
with tf.Session(graph=graph) as sess:
    # Initialize variables
    sess.run(init)

    try:
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored from file: %s" % checkpoint_path)
    except Exception as e:
        print("Model restore failed {}".format(e))

    seed = u"»|5creature|4|6angel|7|8"
    #seed = u"»|5planeswalker|4|6"
    state = np.zeros((NUM_LAYERS,2,1,LSTM_SIZE))
    sample = [seed[0]]
    start = [WH.vocab.char2id(seed[0])]
    for i in range(1,len(seed)):
        s = sess.run(final_state, feed_dict={x: np.array(start).reshape((1,1)), init_state: state, dropout_prob: 1.0})
        start = [WH.vocab.char2id(seed[i])]
        sample.append(seed[i])
        state = s

    #for _ in range(100):
    pred_letter = ""
    while pred_letter != WH.vocab.vocab[-1]:
        s, p = sess.run([final_state, pred], feed_dict={x: np.array(start).reshape((1,1)), init_state: state, dropout_prob: 1.0})
        pred_letter = np.random.choice(WH.vocab.vocab, 1, p=p[0][0])[0]
        pred_id = WH.vocab.char2id(pred_letter)
        start = [pred_id]
        state = s
        sample.append(pred_letter)
        if len(sample) > 500:
            break
    print("SAMPLE: {}".format(''.join(sample)))
