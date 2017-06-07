import numpy as np
import tensorflow as tf
import pickle
import os
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# PATHS
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = os.path.join(DIR_PATH,"saved","conv","model.ckpt")
MODEL_PATH = os.path.join(DIR_PATH,"saved","conv","model_steps.ckpt")
LOG_DIR = "/tmp/tensorflow/log"#os.path.join(DIR_PATH,"tensorboard","conv")
DATA_PATH = os.path.join(DIR_PATH,"data","notMNIST.pkl")


# Set random seed
seed = 36 # Pick your favorite
np.random.seed(seed)
tf.set_random_seed(seed)

if tf.gfile.Exists(LOG_DIR):
    tf.gfile.DeleteRecursively(LOG_DIR)
tf.gfile.MakeDirs(LOG_DIR)

######################################### CONSTANTS ########################################
NUM_CLASSES = 10 # The number of digits present in the dataset
DEVICE = "/gpu:0" # Controls whether we run on CPU or GPU
IMG_SIZE = 28 # Size of an image in dataset
NUM_CHANNELS = 1
LEARNING_RATE = 0.001
GEN_LEARNING_RATE = 0.001

# GENERATOR
GEN_SIZE_IN = 100
GEN_SIZE_1 = 128 # 1st layer number of features
GEN_SIZE_2 = 256 # 2nd layer number of features
GEN_SIZE_3 = IMG_SIZE * IMG_SIZE # final layer
# DISCRIMINATOR
HIDDEN_SIZE_1 = 32 # 1st layer number of features
HIDDEN_SIZE_2 = 64 # 2nd layer number of features
HIDDEN_SIZE_3 = 128 # 3rd layer
HIDDEN_SIZE_4 = 1024 # Dense layer -- ouput
KERNEL_SIZE_1 = [10,10]
KERNEL_SIZE_2 = [5,5]
NUM_CLASSES = 10 # MNIST total classes (0-9 digits)

BATCH_SIZE = 100
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
MAX_STEPS = 100
LOG_FREQUENCY = 1

# Constants describing the training process.
#MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
#NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
#LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
#INITIAL_LEARNING_RATE = 0.1

# LOAD DATA
with open(DATA_PATH,"rb") as f:
    data = pickle.load(f)
TRAIN_DATASET = data["train_dataset"]
TRAIN_LABELS = data["train_labels"]
TEST_DATASET = data["test_dataset"]
TEST_LABELS = data["test_labels"]
VALID_DATASET = data["valid_dataset"]
VALID_LABELS = data["valid_labels"]



class Batcher:
    def __init__(self,dataset,labels):
        self.counter = 0
        self.dataset = dataset
        self.labels = labels

    def nextBatch(self,batch_size):
        batch = self.dataset[self.counter:self.counter+batch_size],self.labels[self.counter:self.counter+batch_size]
        self.counter = (self.counter + batch_size) % len(self.labels)
        return batch


######################################### UTILITY FUNCTIONS ########################################
with tf.device(DEVICE):

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def activation_summary(x):
        """Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations."""
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    # DEFINE MODEL
    graph = tf.Graph()
    with graph.as_default():
        # tf Graph input

        # Input placeholders
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE], name='x-input')
            g = tf.placeholder(tf.float32, [None,GEN_SIZE_IN] , name="generator_input") # Random input vector
            #y = tf.placeholder(tf.int32, [None,], name='y-input') # Labels

        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(x,[-1,IMG_SIZE, IMG_SIZE, 1])
            tf.summary.image('input', image_shaped_input, NUM_CLASSES)
            #shaped_labels = tf.reshape(tf.one_hot(y,NUM_CLASSES),[-1,NUM_CLASSES])

        # We can't initialize these variables to 0 - the network will get stuck.
        def init_weight(shape,name,sd=None):
            """Create a weight variable with appropriate initialization."""
            if not sd:
                sd = 1. / tf.sqrt(shape[0] / 2.)
            initial = tf.truncated_normal(shape, stddev=sd)
            return tf.get_variable(name="{}_weights".format(name), initializer=initial)
            #return tf.Variable(initial)

        def init_bias(shape,name,c=0.1):
            """Create a bias variable with appropriate initialization."""
            initial = tf.constant(c, shape=shape)
            return  tf.get_variable(name="{}_bias".format(name), initializer=initial)
            #return tf.Variable(initial)

        def convLayer(input_tensor, kernel_shape, channel_dim, output_dim, layer_name, act=tf.nn.relu):
            with tf.variable_scope(layer_name) as scope:
                kernel = init_weight(shape=kernel_shape+[channel_dim, output_dim],name=layer_name, sd=5e-2)
                #variable_summaries(kernel)
                conv = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding='SAME')
                biases = init_bias([output_dim], name=layer_name, c=0.0)
                #variable_summaries(biases)
                pre_activation = tf.nn.bias_add(conv, biases)
                tf.summary.histogram('conv_pre_activations', pre_activation)
                conv = act(pre_activation, name=scope.name)
                tf.summary.histogram('conv_activations',conv)
                #image_shaped_conv_first = tf.reshape(kernel,[output_dim * channel_dim] + kernel_shape + [1])
                #tf.summary.image('{}_conv'.format(layer_name), image_shaped_conv_first, 8)
                # Pooling
                pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[3, 3], strides=3)
                return pool

        def linearLayer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.sigmoid):
            with tf.variable_scope(layer_name) as scope:
                # Weights
                weights = init_weight(shape=[input_dim,output_dim],name=layer_name)
                #variable_summaries(weights)
                # Biases
                biases = init_bias([output_dim], name=layer_name, c=0.0)
                #variable_summaries(biases)
                # Pre-Activations
                lin = tf.matmul(input_tensor, weights)
                pre_activation = tf.nn.bias_add(lin, biases)
                tf.summary.histogram('lin_pre_activations', pre_activation)
                # Activations
                activations = act(pre_activation, name=scope.name)
                tf.summary.histogram('lin_activations',activations)
                return activations

        # DEFINE GENERATOR
        def generator(gen_input):
            gen1 = linearLayer(gen_input, GEN_SIZE_IN, GEN_SIZE_1, 'gen_layer1', act=tf.nn.relu)
            gen2 = linearLayer(gen1, GEN_SIZE_1, GEN_SIZE_2, 'gen_layer2', act=tf.nn.relu)
            gen3 = linearLayer(gen2, GEN_SIZE_2, GEN_SIZE_3, 'gen_layer3')
            image_shaped_gen = tf.reshape(gen3,[-1,IMG_SIZE, IMG_SIZE, 1])
            tf.summary.image('generated_input', image_shaped_gen, NUM_CLASSES)
            #return gen2
            return image_shaped_gen

        # DEFINE DISCRIMINATOR
        def discriminatorConv(input_tensor):
            hidden1 = convLayer(input_tensor, KERNEL_SIZE_1, 1, HIDDEN_SIZE_1, 'layer1')
            hidden2 = convLayer(hidden1, KERNEL_SIZE_2, HIDDEN_SIZE_1, HIDDEN_SIZE_2, 'layer2')
            hidden3 = convLayer(hidden2, KERNEL_SIZE_2, HIDDEN_SIZE_2, HIDDEN_SIZE_3, 'layer3')
            # Dense Layer
            flat = tf.reshape(hidden3, [-1, hidden3.get_shape().as_list()[1] * hidden3.get_shape().as_list()[2] * HIDDEN_SIZE_3])
            with tf.variable_scope("dense") as scope:
                dense = tf.layers.dense(inputs=flat, units=HIDDEN_SIZE_4, activation=tf.nn.relu)
                # Logits Layer
                logits = tf.layers.dense(inputs=dense, units=1)
            prob = tf.nn.sigmoid(logits)
            return prob

        def discriminator(input_tensor):
            dis1 = linearLayer(input_tensor,GEN_SIZE_3, GEN_SIZE_1, 'dis_layer1')
            dis2 = linearLayer(dis1, GEN_SIZE_1, 1, 'dis_layer2')
            return dis2


        with tf.variable_scope("generator") as scope:
            fake_data = generator(g)

        with tf.variable_scope("discriminator") as scope:
            fake_prob = discriminatorConv(fake_data)
            scope.reuse_variables()
            real_prob = discriminatorConv(image_shaped_input)
            #real_prob = discriminator(x)

        # Define loss function(s)
        with tf.name_scope('loss'):
            discriminator_loss = -tf.reduce_mean(tf.log(real_prob) + tf.log(1. - fake_prob))
            generator_loss = -tf.reduce_mean(tf.log(fake_prob))
            tf.summary.scalar('discriminator_loss', discriminator_loss)
            tf.summary.scalar('generator_loss', generator_loss)

        # Define optimizer
        with tf.name_scope('train'):
            # Only update the variables associated with each network
            #   If we update the discriminator while optimizing the generator it will lose the ability to discriminate
            #   and our generator will no longer have an adversary.
            #   The same is true of the generator.
            train_d_step = tf.train.AdamOptimizer().minimize(discriminator_loss,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator'))
            train_g_step = tf.train.AdamOptimizer().minimize(generator_loss,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'))

        # # Define and track accuracy
        # with tf.name_scope('accuracy'):
        #     with tf.name_scope('correct_prediction'):
        #         correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(shaped_labels, 1))
        #     with tf.name_scope('accuracy'):
        #         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # tf.summary.scalar('accuracy', accuracy)

        # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'test'))

        # Initializing the variables
        init = tf.global_variables_initializer()

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()

    #Running first session
    #def main():
    with tf.Session(graph=graph) as sess:#,config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Initialize variables
        sess.run(init)

        try:
            ckpt = tf.train.get_checkpoint_state(SAVE_PATH)
            saver.restore(sess, ckpt.model_checkpoint_path)
            #saver.restore(sess, SAVE_PATH)
            print("Model restored from file: %s" % SAVE_PATH)
        except Exception as e:
            print("Model restore failed {}".format(e))

        #train_batcher = Batcher(TRAIN_DATASET,TRAIN_LABELS)
        #total_batch = int(len(TRAIN_DATASET)/BATCH_SIZE)

        total_batch = int(mnist.train.num_examples/BATCH_SIZE)

        # Training cycle
        for epoch in range(MAX_STEPS):
            avg_cost = 0.

            # Loop over all batches
            for i in range(total_batch):
                #batch_x, batch_y = train_batcher.nextBatch(BATCH_SIZE)
                batch_x, batch_y  = mnist.train.next_batch(BATCH_SIZE)
                G_INPUT = np.random.uniform(-1., 1., size=[BATCH_SIZE,GEN_SIZE_IN])
                # Run optimization op (backprop) and cost op (to get loss value)
                summary,g_unused, _d,_g = sess.run([merged, fake_data, train_d_step, train_g_step], feed_dict={x: batch_x, g: G_INPUT})
                train_writer.add_summary(summary, i)

                # Keep track of meta data
                if i % 100 == 0:
                    #   I'm not a billion percent sure what this does....
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _d,_g = sess.run([merged, train_d_step, train_g_step],
                                          feed_dict={x: batch_x, g: G_INPUT},
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, "step_{}_{}".format(epoch,i))
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)

            # Display logs per epoch step
            if epoch % LOG_FREQUENCY == 0:
                #G_TEST = np.array(np.random.randint(NUM_CLASSES,size=len(TEST_DATASET)),dtype="int32")
                #G_TEST = np.array(np.random.randint(NUM_CLASSES,size=len(mnist.test.images)),dtype="int32")
                #summary = sess.run([merged], feed_dict={x: TEST_DATASET, y: TEST_LABELS, g: G_TEST})
                #summary,_g = sess.run([merged, fake_data], feed_dict={x: mnist.test.images, g: G_TEST})
                #test_writer.add_summary(summary, i)
                save_path = saver.save(sess, MODEL_PATH, global_step = epoch)
                print('Step %s' % epoch)
        # Cleanup
        train_writer.close()
        test_writer.close()
        print("Training Finished!")

        # Save model weights to disk
        save_path = saver.save(sess, SAVE_PATH)
        print("Model saved in file: %s" % save_path)
