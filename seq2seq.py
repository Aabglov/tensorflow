import numpy as np
import tensorflow as tf
from helpers.utils import save,load
import os
import time
from helpers import word_helpers, rap_helper
import caffeine

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# PATHS
SAVE_DIR = "seq2seq"
CHECKPOINT_NAME = "rap_seq_steps.ckpt"
DATA_NAME = "ohhla.txt"
PICKLE_PATH = "seq_rh.pkl"
SUBDIR_NAME = "seq2seq"

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path,"saved",SAVE_DIR,CHECKPOINT_NAME)
save_path = os.path.join(dir_path,"saved",SAVE_DIR)
checkpoint_path = os.path.join(save_path,"checkpoint")
data_path = os.path.join(dir_path,"data",SUBDIR_NAME,DATA_NAME)
LOG_DIR = "/tmp/tensorflow/log"


# Set random seed
seed = 36 # Pick your favorite
np.random.seed(seed)
tf.set_random_seed(seed)

######################################### CONSTANTS ########################################
DEVICE = "/gpu:0" # Controls whether we run on CPU or GPU
LEARNING_RATE = 3e-3 #2e-3
DECAY_RATE = 1.0
ADAM_BETA = 0.5
GRAD_CLIP = 5.0

BATCH_SIZE = 1#64
MAX_STEPS = 1000
LOG_FREQUENCY = 100
# max length of 100 excludes about 1200 of 300k samples.
# max length of 200 excludes about 100 of 300k
MAX_SEQ_LEN = 50
EMBEDDING_DIM = 128
LSTM_SIZE = 128 #512
NUM_LAYERS = 2 #3
PRIME_TEXT = "u'\xbb'Here we go"
DEBUG = False #True


if tf.gfile.Exists(LOG_DIR):
    tf.gfile.DeleteRecursively(LOG_DIR)
tf.gfile.MakeDirs(LOG_DIR)

##################################################################################################
######################## DATA PREP ########################
SB = rap_helper.getRapData(os.path.join(save_path,PICKLE_PATH),max_seq_len=MAX_SEQ_LEN)
songs = SB.songs
vocab_obj = SB.vocab
vocab = vocab_obj.vocab
SS = rap_helper.SongSequencer(songs,vocab,MAX_SEQ_LEN)

N_CLASSES = len(vocab)

vocab_lookup = SS.vocab_lookup
reverse_vocab_lookup = SS.reverse_vocab_lookup

NUM_SAMPLES = SS.num_songs

for i in range(len(vocab)):
    char = vocab[i]
    vocab_lookup[char] = i
    reverse_vocab_lookup[i] = char

pad_val = vocab_lookup[rap_helper.PAD]

encoder_input,decoder_input,decoder_target = SS.__next__()


# [30002 30002 30002 30002 30002 30002 30002 30002 30002 30002 30002 30002
#  30002 30002 30002 30002 30002 30002 30002    36    18   110    22   934
#      4   948   948 30001 30001    12  3732  7908    40   428    80  3799
#  19730   985   495 30020   948    54    32     5 23544     0   948   948
#    188     0]
# ¬ ¬ ¬ ¬ ¬ ¬ ¬ ¬ ¬ ¬ ¬ ¬ ¬ ¬ ¬ ¬ ¬ ¬ ¬ can we make this quick ?   ¿ ¿ and andrew barrett are having an incredibly horrendous public break -  up on the quad .   again .



######################################### UTILITY FUNCTIONS ########################################
#with tf.device(DEVICE):
def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1)*s)))

# Input placeholders
with tf.name_scope('input'):
    # Placeholders
    en_input = tf.placeholder(tf.int32, [None, MAX_SEQ_LEN], name='encoder_input')
    #en_target = tf.placeholder(tf.int32, [None, MAX_SEQ_LEN], name='encoder_target')
    dec_input = tf.placeholder(tf.int32, [None, MAX_SEQ_LEN], name='decoder_input')
    dec_target = tf.placeholder(tf.int32, [None, MAX_SEQ_LEN], name='decoder_target')
    training = tf.placeholder(tf.int32, shape=[], name="training_switch")
    # Our initial state placeholder:
    init_state_placeholder = tf.placeholder(tf.float32, [NUM_LAYERS, 2, None, LSTM_SIZE], name='encoder_state_placeholder')
    embedding = tf.get_variable("embedding", [N_CLASSES, LSTM_SIZE],dtype=tf.float32)

# Recurrent Neural Network
def RNN(input_tensor,init_state,num_layers,lstm_size,name,reuse_flag=False,dropout_prob=0.3):
    with tf.variable_scope(name, reuse=reuse_flag):
        # Create appropriate LSTMStateTuple for dynamic_rnn function out of our placeholder
        l = tf.unstack(init_state, axis=0)

        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1]) for idx in range(num_layers)]
        )

        # RNN
        lstm = tf.contrib.rnn.LSTMCell(lstm_size)
        dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_prob)
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([dropout] * num_layers)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=stacked_lstm,
                                                     inputs=input_tensor,
                                                     initial_state=rnn_tuple_state)#stacked_lstm.zero_state(batch_size,tf.float32))
        return rnn_outputs,final_state

# # Decoder -- uses REUSE
# def DecoderRNN(input_tensor,init_state,num_layers,lstm_size,name,dropout_prob=0.3):
#     with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#         # Create appropriate LSTMStateTuple for dynamic_rnn function out of our placeholder
#         l = tf.unstack(init_state, axis=0)
#
#         rnn_tuple_state = tuple(
#             [tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1]) for idx in range(num_layers)]
#         )
#
#         # RNN
#         lstm = tf.contrib.rnn.LSTMCell(lstm_size)
#         dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_prob)
#         stacked_lstm = tf.contrib.rnn.MultiRNNCell([dropout] * num_layers)
#         rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=stacked_lstm,
#                                                      inputs=input_tensor,
#                                                      initial_state=rnn_tuple_state)#stacked_lstm.zero_state(batch_size,tf.float32))
#         return rnn_outputs,final_state

def embed(input_tensor):
    with tf.name_scope('embedding'):
        rnn_inputs = tf.nn.embedding_lookup(embedding, input_tensor)
    return rnn_inputs

# ENCODER
def encoder(input_tensor,init_state):
    num_layers = NUM_LAYERS
    lstm_size = LSTM_SIZE
    num_classes = len(vocab)
    dropout_prob = 0.3

    rnn_inputs = embed(input_tensor)

    #Inputs
    with tf.name_scope('encoder'):
        rnn_out,rnn_state = RNN(input_tensor=rnn_inputs,
                                init_state=init_state,
                                num_layers=num_layers,
                                lstm_size=lstm_size,
                                name='encoder',
                                reuse_flag=False,
                                dropout_prob=dropout_prob)

    return rnn_out,rnn_state

# DECODER
def decoder(input_tensor,init_state,train):
    num_layers = NUM_LAYERS
    lstm_size = LSTM_SIZE
    dropout_prob = 0.3

    embedded_inputs = embed(input_tensor)
    unstacked_rnn_inputs = tf.unstack(tf.reshape(embedded_inputs,[-1,MAX_SEQ_LEN,1,lstm_size]), axis=1)
    rnn_input = unstacked_rnn_inputs[0]
    #rnn_input = input_tensor
    state = init_state
    outs = []

    def getNextDecIn(i):
        return unstacked_rnn_inputs[i]

    # Pass the initial character (first entry)
    # through the decoder as a separate process.
    # The decoder always needs a seed character to perform the
    # first output.  Then depending on whether we're trainging or
    # prediction (actual use of the model)
    # we will either reuse the output as the next input or
    # get the next seed character from the decoder_input and perform
    # the process again.
    # decoder_output,state = DecoderRNN(rnn_input,state,num_layers,lstm_size,'decoder',dropout_prob)
    decoder_output,state = RNN(input_tensor=rnn_input,
                            init_state=state,
                            num_layers=num_layers,
                            lstm_size=lstm_size,
                            name='decoder',
                            reuse_flag=tf.AUTO_REUSE,
                            dropout_prob=dropout_prob)

    outs.append(tf.reshape(decoder_output,[-1,lstm_size]))

    def getDecOut():
        return decoder_output

    with tf.name_scope('decoder'):
        for _i in range(1,MAX_SEQ_LEN):
            # Why do conditionals have to suck so hard in
            # Machine Learning modules?
            # It's not as bad as Theano, but still......
            rnn_input = tf.cond(train > 0,lambda: getNextDecIn(_i),lambda: getDecOut())
            # decoder_output,state = DecoderRNN(rnn_input,state,num_layers,lstm_size,'decoder',dropout_prob)
            decoder_output,state = RNN(input_tensor=rnn_input,
                                    init_state=state,
                                    num_layers=num_layers,
                                    lstm_size=lstm_size,
                                    name='decoder',
                                    reuse_flag=tf.AUTO_REUSE,
                                    dropout_prob=dropout_prob)

            outs.append(tf.reshape(decoder_output,[-1,lstm_size]))
    total_out = tf.stack(outs,axis=1)
    dense_out = tf.layers.dense(total_out,len(vocab),activation=None,use_bias=True)
    return dense_out

with tf.variable_scope("model") as scope:
    encoder_out,encoder_state = encoder(en_input,init_state_placeholder)
    # decoder_out are our logits
    decoder_out = decoder(dec_input, encoder_state, training)
    pred = tf.nn.softmax(decoder_out)

# Define loss function(s)
with tf.name_scope('loss'):
    dec_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=dec_target, logits=decoder_out)
    cost = tf.reduce_mean(dec_loss)
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
train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'), tf.get_default_graph())

# Initializing the variables
init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

#Running first session
#def main():
if __name__ == "__main__":
    with tf.Session() as sess:#,config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Initialize variables
        sess.run(init)
        sess.run(local_init)

        try:
            ckpt = tf.train.get_checkpoint_state(save_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored from file: %s" % save_path)
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
                # # Generate a batch
                # batch_x =  encoder_input[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                # batch_y = decoder_target[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

                encoder_input,decoder_input,decoder_target = SS.__next__()

                # Run optimization op (backprop) and cost op (to get loss value)
                fd = {en_input: encoder_input,
                    dec_input: decoder_input,
                    dec_target: decoder_target,
                    init_state_placeholder: new_state,
                    training:1}

                summary, s, c, _ = sess.run([merged, encoder_state, cost, optimizer], feed_dict=fd)
                train_writer.add_summary(summary, epoch)

                sample_batch = SS.arrayify(SS.padSequence(PRIME_TEXT))
                unused_y = np.zeros((1,MAX_SEQ_LEN)).astype(np.int64)
                sample_decoder_input = SS.arrayify([vocab_obj.split_char]* MAX_SEQ_LEN)
                sample_init_state = np.zeros((NUM_LAYERS,2,1,LSTM_SIZE))

                # print(sample_batch.shape,encoder_input.shape)
                # print(sample_decoder_input.shape,decoder_input.shape)
                # print(unused_y.shape,decoder_target.shape)
                # print(new_state.shape,sample_init_state.shape)
                #
                # print(sample_batch.dtype,encoder_input.dtype)
                # print(sample_decoder_input.dtype,decoder_input.dtype)
                # print(unused_y.dtype,decoder_target.dtype)
                # print(new_state.dtype,sample_init_state.dtype)

                fd = {en_input: sample_batch,
                    dec_input: sample_decoder_input,
                    dec_target: unused_y,
                    init_state_placeholder: sample_init_state,
                    training:0}

                summary, predicted_output = sess.run([merged, pred], feed_dict=fd)

                first_pred_output = predicted_output[0]
                pred_letters = []
                for p in first_pred_output:
                    #pred_letter = np.random.choice(vocab, 1, p=p)[0]
                    pred_letter = reverse_vocab_lookup[weighted_pick(p)]
                    #pred_letter = reverse_vocab_lookup[np.argmax(p)]
                    pred_letters.append(pred_letter)
                sample = ''.join(pred_letters)
                print('iteration',i,'of',NUM_SAMPLES//BATCH_SIZE,'cost: ', c, 'pred: ',sample)


                # Display logs per epoch step
                if i % LOG_FREQUENCY == 0:
                    #   I'm not a billion percent sure what this does....
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    fd= {en_input: encoder_input,
                        dec_input: decoder_input,
                        dec_target: decoder_target,
                        init_state_placeholder: new_state,
                        training:1} # training doesn't matter here
                    summary, s, c, _ = sess.run([merged, encoder_state, cost, optimizer],
                                          feed_dict=fd,
                                          options=run_options,
                                          run_metadata=run_metadata)
                    current_iter = (epoch*NUM_SAMPLES) + i
                    train_writer.add_run_metadata(run_metadata, "step_{}".format(current_iter))
                    train_writer.add_summary(summary, current_iter)
                    print('Adding run metadata for', current_iter)
                    save_path = saver.save(sess, model_path, global_step = current_iter)
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
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
