import numpy as np
import tensorflow as tf
from utils import save,load
import os
import time
import word_helpers
import dialog_parser

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# PATHS
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = os.path.join(DIR_PATH,"saved","text","model.ckpt")
CHKPT_PATH = os.path.join(DIR_PATH,"saved","text")
LOG_DIR = "/tmp/tensorflow/log"
#DATA_PATH = os.path.join(DIR_PATH,"data","cards_tokenized.txt")
LOAD_PATH = os.path.join(DIR_PATH,"data","dialog")

# Set random seed
seed = 36 # Pick your favorite
np.random.seed(seed)
tf.set_random_seed(seed)

######################################### CONSTANTS ########################################
DEVICE = "/gpu:0" # Controls whether we run on CPU or GPU
LEARNING_RATE = 0.0002 #0.001
DECAY_RATE = 1.0
ADAM_BETA = 0.5
GRAD_CLIP = 5.0

BATCH_SIZE = 64
MAX_STEPS = 40000
LOG_FREQUENCY = 100
# max length of 100 excludes about 1200 of 300k samples.
# max length of 200 excludes about 100 of 300k
MAX_SEQ_LEN = 50
EMBEDDING_DIM = 128
LSTM_SIZE = 128 #512
NUM_LAYERS = 2 #3

DEBUG = False #True

GO =    dialog_parser.GO
UNK =   dialog_parser.UNK
PAD =   dialog_parser.PAD
EOS =   dialog_parser.EOS
SPLIT = dialog_parser.SPLIT

if tf.gfile.Exists(LOG_DIR):
    tf.gfile.DeleteRecursively(LOG_DIR)
tf.gfile.MakeDirs(LOG_DIR)

###################################################################################################

def padSequence(seq,pad_char):
    if len(seq) >= MAX_SEQ_LEN:
        return seq[:MAX_SEQ_LEN]
    else:
        pad_num = MAX_SEQ_LEN - len(seq)
        return ([pad_char] * pad_num) + seq

try:
    input_seq = load(LOAD_PATH,"inputs.pkl")
    target_seq = load(LOAD_PATH,"targets.pkl")
    convs = load(LOAD_PATH,"convs.pkl")
    vocab = load(LOAD_PATH,"vocab.pkl")
    print("Loaded prepared data...")

    # save(input_seq,"inputs.pkl",protocol=2)
    # save(target_seq,"targets.pkl",protocol=2)
    # save(convs,"convs.pkl",protocol=2)
    # save(vocab,"vocab.pkl",protocol=2)


except Exception as e:
    print("FAILED TO LOAD:")
    print(e)

    input_seq,target_seq,convs,vocab = dialog_parser.parseDialog()

    save(input_seq,LOAD_PATH,"inputs.pkl")
    save(target_seq,LOAD_PATH,"targets.pkl")
    save(convs,LOAD_PATH,"convs.pkl")
    save(vocab,LOAD_PATH,"vocab.pkl")



######################## DATA PREP ########################
vocab_lookup = {}
reverse_vocab_lookup = {}

NUM_SAMPLES = len(input_seq)

for i in range(len(vocab)):
    word = vocab[i]
    vocab_lookup[word] = i
    reverse_vocab_lookup[i] = word

pad_val = vocab_lookup[PAD]

if DEBUG:
    print("vocab length: ",len(vocab))
    #print(input_seq[1])
    print(input_seq[0])
    print(target_seq[0])
    print(" ".join([reverse_vocab_lookup[i] for i in input_seq[0]]))
    print(" ".join([reverse_vocab_lookup[i] for i in target_seq[0]]))
    #print(convs[0].lines[1])

    totes = len(input_seq)
    less_than_max = 0
    for i in input_seq:
        if len(i) < MAX_SEQ_LEN:
            less_than_max += 1
    print(totes)
    print(less_than_max)
    print((less_than_max)/totes)

    print(padSequence(input_seq[0],pad_val))

padded_input = [padSequence(i,pad_val) for i in input_seq]
padded_target= [padSequence(t,pad_val) for t in target_seq]

encoder_input = np.asarray(padded_input,dtype='float32')
decoder_target = np.asarray(padded_target,dtype='float32')


if DEBUG:
    print(encoder_input.shape)
    print(decoder_target.shape)


######################################### UTILITY FUNCTIONS ########################################
with tf.device(DEVICE):

    # DEFINE MODEL
    graph = tf.Graph()
    with graph.as_default():

        # Input placeholders
        with tf.name_scope('input'):
            # Placeholders
            x = tf.placeholder(tf.int32, [None, MAX_SEQ_LEN], name='input_placeholder')
            y = tf.placeholder(tf.int32, [None, MAX_SEQ_LEN], name='labels_placeholder')
            #dropout_prob = tf.placeholder(tf.float32)
            # Our initial state placeholder:
            init_state_placeholder = tf.placeholder(tf.float32, [NUM_LAYERS, 2, None, LSTM_SIZE], name='encoder_state_placeholder')

        # Recurrent Neural Network
        def RNN(input_tensor,init_state,num_layers,lstm_size,name,dropout_prob=0.3):
            with tf.variable_scope(name):
                # Create appropriate LSTMStateTuple for dynamic_rnn function out of our placeholder
                l = tf.unstack(init_state, axis=0)
                rnn_tuple_state = tuple(
                    [tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1]) for idx in range(num_layers)]
                )

                # Get dynamic batch_size
                batch_size = tf.shape(x)[0]

                # RNN
                lstm = tf.contrib.rnn.LSTMCell(lstm_size)
                dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_prob)
                stacked_lstm = tf.contrib.rnn.MultiRNNCell([dropout] * num_layers)
                rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=stacked_lstm,
                                                             inputs=input_tensor,
                                                             initial_state=rnn_tuple_state)#stacked_lstm.zero_state(batch_size,tf.float32))
                return rnn_outputs,final_state


        # ENCODER
        def encoder(input_tensor,init_state):
            num_layers = NUM_LAYERS
            lstm_size = LSTM_SIZE
            num_classes = len(vocab)
            dropout_prob = 0.3

            #Inputs
            with tf.name_scope('encoder'):
                embedding = tf.get_variable("embedding", [num_classes, EMBEDDING_DIM])
                rnn_inputs = tf.nn.embedding_lookup(embedding, input_tensor)

            rnn_out,rnn_state = RNN(rnn_inputs,init_state,num_layers,lstm_size,'encoder',dropout_prob)

            return rnn_out,rnn_state

        # DECODER
        def decoder(input_tensor,init_state):
            num_layers = NUM_LAYERS
            lstm_size = LSTM_SIZE
            dropout_prob = 0.3
            rnn_out,rnn_state = RNN(input_tensor,init_state,num_layers,lstm_size,'decoder',dropout_prob)

            dense_out = tf.layers.dense(rnn_out,len(vocab),activation=None,use_bias=True)

            return dense_out

        with tf.variable_scope("model") as scope:
            encoder_out,encoder_state = encoder(x,init_state_placeholder)
            # decoder_out are our logits
            decoder_out = decoder(encoder_out, encoder_state)
            pred = tf.nn.softmax(decoder_out)

        # Define loss function(s)
        with tf.name_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=decoder_out)
            cost = tf.reduce_mean(loss)
            tf.summary.scalar('cost', cost)

        # Define optimizer
        with tf.name_scope('train'):
            #train_d_step = tf.train.AdamOptimizer(DIS_LEARNING_RATE,beta1=ADAM_BETA).minimize(discriminator_loss,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator'))
            #train_step = tf.train.AdamOptimizer(LEARNING_RATE,beta1=ADAM_BETA).minimize(loss)
            lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), GRAD_CLIP)
            with tf.name_scope('optimizer'):
                op = tf.train.AdamOptimizer(lr)
            optimizer = op.apply_gradients(zip(grads, tvars))

        # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph)

        # Initializing the variables
        init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()

#Running first session
#def main():
with tf.Session(graph=graph) as sess:#,config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Initialize variables
    sess.run(init)
    sess.run(local_init)

    try:
        ckpt = tf.train.get_checkpoint_state(CHKPT_PATH)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored from file: %s" % SAVE_PATH)
    except Exception as e:
        print("Model restore failed {}".format(e))

    # Training cycle
    already_trained = 0
    for epoch in range(already_trained,already_trained+MAX_STEPS):
        # Set learning rate
        sess.run(tf.assign(lr,LEARNING_RATE * (DECAY_RATE ** epoch)))
        for i in range(0,NUM_SAMPLES//BATCH_SIZE):
            # Reset state value
            new_state = np.zeros((NUM_LAYERS,2,BATCH_SIZE,LSTM_SIZE))
            # Generate a batch
            batch_x =  encoder_input[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            batch_y = decoder_target[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            #print(batch_x.shape)
            #print(batch_y.shape)
            #print(batch_x[0][-10:])
            #HODOR

            # Run optimization op (backprop) and cost op (to get loss value)
            fd= {x: batch_x, y: batch_y, init_state_placeholder: new_state}
            summary, s, predicted_output, c, _ = sess.run([merged, encoder_state, pred, cost, optimizer], feed_dict=fd)
            train_writer.add_summary(summary, epoch)

            first_pred_output = predicted_output[0]
            pred_letters = []
            for p in first_pred_output:
                #pred_letter = np.random.choice(vocab, 1, p=p)[0]
                pred_letter = reverse_vocab_lookup[np.argmax(p)]
                pred_letters.append(pred_letter)
            sample = ' '.join(pred_letters)
            print('cost: ', c, 'pred: ',sample)


        # Display logs per epoch step
        if epoch % LOG_FREQUENCY == 0:
            #   I'm not a billion percent sure what this does....
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            fd= {x: batch_x, y: batch_y, init_state_placeholder: new_state}
            summary, s, c, _ = sess.run([merged, encoder_state, cost, optimizer],
                                  feed_dict=fd,
                                  options=run_options,
                                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, "step_{}".format(epoch))
            train_writer.add_summary(summary, epoch)
            print('Adding run metadata for', epoch)
            save_path = saver.save(sess, SAVE_PATH, global_step = epoch)
            print('Step %s' % epoch)

    # Cleanup
    #   Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
    #   Close writers
    train_writer.close()
    #test_writer.close()
    print("Training Finished!")

    # Save model weights to disk
    save_path = saver.save(sess, SAVE_PATH)
    print("Model saved in file: %s" % save_path)
