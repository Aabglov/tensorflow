import numpy as np
import tensorflow as tf
import pickle
import os
import time

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# PATHS
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = os.path.join(DIR_PATH,"saved","conv","model.ckpt")
CHKPT_PATH = os.path.join(DIR_PATH,"saved","conv")
LOG_DIR = "/tmp/tensorflow/log"#os.path.join(DIR_PATH,"tensorboard","conv")
DATA_PATH = os.path.join(DIR_PATH,"data","celeb")


# Set random seed
seed = 36 # Pick your favorite
np.random.seed(seed)
tf.set_random_seed(seed)

######################################### CONSTANTS ########################################
NUM_CLASSES = 10 # The number of digits present in the dataset
DEVICE = "/gpu:0" # Controls whether we run on CPU or GPU
NUM_CHANNELS = 3
DIS_LEARNING_RATE = 0.0002 #0.001
GEN_LEARNING_RATE = 0.0002 #0.001
ADAM_BETA = 0.5

# GENERATOR
GEN_SIZE_1 = 512 #512 # 1st layer number of features
GEN_SIZE_2 = 256  #256 # 2nd layer number of features
GEN_SIZE_3 = 128 #128 # 3rd layer
GEN_SIZE_4 = 64 #64 # final layer


# DISCRIMINATOR
HIDDEN_SIZE_1 = 32 # 1st layer number of features
HIDDEN_SIZE_2 = 64 # 2nd layer number of features
HIDDEN_SIZE_3 = 128 # 3rd layer
HIDDEN_SIZE_4 = 256 # Dense layer -- ouput
DISC_KERNEL = [5,5]

BATCH_SIZE = 50
MAX_STEPS = 40000
LOG_FREQUENCY = 100


if tf.gfile.Exists(LOG_DIR):
    tf.gfile.DeleteRecursively(LOG_DIR)
tf.gfile.MakeDirs(LOG_DIR)

######################################### UTILITY FUNCTIONS ########################################
with tf.device(DEVICE):

    # DEFINE MODEL
    graph = tf.Graph()
    with graph.as_default():

        # Input placeholders
        with tf.name_scope('input'):
            # Placeholders
            x = tf.placeholder(tf.int32, [None, NUM_STEPS], name='input_placeholder')
            y = tf.placeholder(tf.int32, [None, NUM_STEPS], name='labels_placeholder')
            dropout_prob = tf.placeholder(tf.float32)
            # Our initial state placeholder:
            gen_init_state = tf.placeholder(tf.float32, [num_layers, 2, None, lstm_size], name='gen_state_placeholder')
            dis_init_state = tf.placeholder(tf.float32, [num_layers, 2, None, lstm_size], name='dis_state_placeholder')

        # Recurrent Neural Network
        def RNN(input_tensor,init_state,num_layers,lstm_size,num_classes,dropout_prob):

            # Create appropriate LSTMStateTuple for dynamic_rnn function out of our placeholder
            l = tf.unstack(init_state, axis=0)
            rnn_tuple_state = tuple(
                [tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1]) for idx in range(num_layers)]
            )

            # Get dynamic batch_size
            batch_size = tf.shape(x)[0]

            #Inputs
            embedding = tf.get_variable("embedding", [num_classes, lstm_size])
            rnn_inputs = tf.nn.embedding_lookup(embedding, input_tensor)

            # RNN
            lstm = tf.contrib.rnn.LSTMCell(lstm_size)
            dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_prob)
            stacked_lstm = tf.contrib.rnn.MultiRNNCell([dropout] * num_layers)
            rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=stacked_lstm,
                                                         inputs=rnn_inputs,
                                                         initial_state=rnn_tuple_state)#stacked_lstm.zero_state(batch_size,tf.float32))
            return rnn_outputs,final_state


        # DEFINE GENERATOR USING DECONVOLUTION
        def generator(gen_in):
            # Conceptual explanation:
            # The Generator's job is to try to fool the
            # discriminator by producing convincing, but
            # fake data.
            # It uses a RNN to predict one letter at a time
            # in order to produce a whole sample (multiple words).
            num_layers = 3
            lstm_size = 128
            num_classes = WH.vocab.vocab_size
            dropout_prob = 0.3
            out,state = RNN(gen_in,gen_init_state,num_layers,lstm_size,num_classes,dropout_prob)
            return out

        # DEFINE DISCRIMINATOR
        def discriminator(input_tensor):
            # TODO
            # Implement discriminator
            # The Discriminator's job is to not get fooled
            # by the Generator.
            # I'm not sure how it's going to do this.
            # It needs to take in a text sample (multiple words)
            # and reduce it down to a single classifier
            # that represents whether the sample is legitimate or not.
            # There are a lot of ways to reduce a vector of known size
            # (the size is known because I'm using padding)
            # into a single number, but what method will capture the
            # properties I'm looking for?
            #
            # I need to at least consider the order of letters
            # which means I need some kind of recurrent discriminator.
            # I think I can use a normal RNN for this and only
            # perform analysis on the final state variable.
            #
            # The predicted outputs are useless in the discriminator
            # since we know the correct outputs for any given sample.
            # However, the state variable keeps track of the internal
            # consistency of the sample which means we should be able
            # to feed it through some linear layers and/or softmax
            # layers to get a reliable determination.
            #

            num_layers = 3
            lstm_size = 128
            num_classes = WH.vocab.vocab_size
            dropout_prob = 0.3
            out,state = RNN(input_tensor,dis_init_state,num_layers,lstm_size,num_classes,dropout_prob)

            with tf.variable_scope('softmax'):
                W = tf.get_variable('W', [lstm_size, num_classes])
                b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
            logits = tf.reshape(
                        tf.matmul(tf.reshape(state, [-1, lstm_size]), W) + b,
                        [-1,batch_size, num_classes])
            pred = tf.nn.softmax(logits)

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            cost = tf.reduce_sum(losses)

            return prob, logits


        with tf.variable_scope("generator") as scope:
            fake_data = generatorDeconv(g_shaped)
            #fake_data = generatorSubpixel(g_shaped)

        with tf.variable_scope("discriminator") as scope:
            fake_prob,fake_logits = discriminatorConv(fake_data)
            #fake_prob = discriminator(fake_data)
            scope.reuse_variables()
            real_prob,real_logits = discriminatorConv(image_shaped_input)
            #real_prob = discriminator(x)

        # Define loss function(s)
        with tf.name_scope('loss'):
            discriminator_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_prob)))
            discriminator_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_prob)))
            discriminator_loss = discriminator_loss_real + discriminator_loss_fake
            generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_prob)))
            #discriminator_loss = -tf.reduce_mean(tf.log(real_prob) + tf.log(1. - fake_prob))
            #generator_loss = -tf.reduce_mean(tf.log(fake_prob))
            tf.summary.scalar('discriminator_loss', discriminator_loss)
            tf.summary.scalar('generator_loss', generator_loss)

        # Define optimizer
        with tf.name_scope('train'):
            # Only update the variables associated with each network
            #   If we update the discriminator while optimizing the generator it will lose the ability to discriminate
            #   and our generator will no longer have an adversary.
            #   The same is true of the generator.
            train_d_step = tf.train.AdamOptimizer(DIS_LEARNING_RATE,beta1=ADAM_BETA).minimize(discriminator_loss,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator'))
            train_g_step = tf.train.AdamOptimizer(GEN_LEARNING_RATE,beta1=ADAM_BETA).minimize(generator_loss,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'))

        # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph)
        #test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'test'))

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

        #train_batcher = Batcher(TRAIN_DATASET,TRAIN_LABELS)
        #total_batch = int(len(TRAIN_DATASET)/BATCH_SIZE)
        #Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Training cycle
        for epoch in range(1,MAX_STEPS):


            G_INPUT = np.random.uniform(-1., 1., size=[BATCH_SIZE,GEN_SIZE_IN])
            # Run optimization op (backprop) and cost op (to get loss value)
            summary,img, g_unused, _d,_g = sess.run([merged, image, fake_data, train_d_step, train_g_step], feed_dict={g: G_INPUT})
            train_writer.add_summary(summary, epoch)
            print('Adding run data for', epoch)


            # Display logs per epoch step
            if epoch % LOG_FREQUENCY == 0:
                #   I'm not a billion percent sure what this does....
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, img, _d,_g = sess.run([merged, image, train_d_step, train_g_step],
                                      feed_dict={g: G_INPUT},
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
