import numpy as np
import tensorflow as tf
import pickle
import os
import time

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# PATHS
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = os.path.join(DIR_PATH,"saved","test","model.ckpt")
LOG_DIR = "/tmp/tensorflow/log"#os.path.join(DIR_PATH,"tensorboard","conv")
DATA_PATH = os.path.join(DIR_PATH,"data","celeb")


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
NUM_CHANNELS = 3
LEARNING_RATE = 0.01
IMG_SIZE1 = 218
IMG_SIZE2 = 178


# GENERATOR
GEN_SIZE_IN = 1
GEN_SIZE_1 = 256 # 1st layer number of features
GEN_SIZE_2 = 1000 # 2nd layer number of features
GEN_SIZE_3 = IMG_SIZE1 * IMG_SIZE2 # final layer


BATCH_SIZE = 1
MAX_STEPS = 1000
LOG_FREQUENCY = 100


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
        # Make a queue of file names including all the JPEG images files in the relative
        # image directory.
        filename_queue = tf.train.string_input_producer([os.path.join(DATA_PATH,"000666.jpg")])

        # Read an entire image file which is required since they're JPEGs, if the images
        # are too large they could be split in advance to smaller files or use the Fixed
        # reader to split up the file.
        image_reader = tf.WholeFileReader()


        # Read a whole file from the queue
        filename, image_file = image_reader.read(filename_queue)

        # Decode the image as a JPEG file, this will turn it into a Tensor which we can
        # then use in training.
        image = tf.image.decode_jpeg(image_file)

        # Input placeholders
        with tf.name_scope('input'):
            #x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE], name='x-input')
            y = tf.image.per_image_standardization(image)
            g = tf.placeholder(tf.float32, [None, GEN_SIZE_IN] , name="generator_input") # Random input vector

        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(y,[-1,IMG_SIZE1, IMG_SIZE2, NUM_CHANNELS])
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

        # DEFINE GENERATOR
        def generator(gen_input):
            gen1 = tf.layers.dense(inputs=gen_input, units=GEN_SIZE_1, activation=tf.nn.sigmoid)
            #gen2 = tf.layers.dense(inputs=gen1,      units=GEN_SIZE_2, activation=tf.nn.sigmoid)
            gen3 = tf.layers.dense(inputs=gen1,      units=GEN_SIZE_3 * NUM_CHANNELS, activation=tf.identity)
            image_shaped_gen = tf.reshape(gen3,[-1,IMG_SIZE1, IMG_SIZE2, NUM_CHANNELS])
            tf.summary.image('generated_input', image_shaped_gen, 1)
            #return gen2
            return image_shaped_gen

        with tf.variable_scope("generator") as scope:
            pred = generator(g)


        # Define loss function
        with tf.name_scope('loss'):
            loss = tf.losses.mean_squared_error(image_shaped_input,pred)
        tf.summary.scalar('loss',loss)

        # Define optimizer
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        # Define and track accuracy
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(pred, image_shaped_input)
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

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
            print("Model restored from file: %s" % SAVE_PATH)
        except Exception as e:
            print("Model restore failed {}".format(e))

        #Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Training cycle
        for epoch in range(MAX_STEPS):
            avg_cost = 0.

            #batch_x, batch_y = train_batcher.nextBatch(BATCH_SIZE)
            #batch_x, batch_y  = mnist.train.next_batch(BATCH_SIZE)
            G_INPUT = np.array([[0.5]])
            summary, _ = sess.run([merged, train_step], feed_dict={g: G_INPUT})
            train_writer.add_summary(summary, epoch)
            print('Adding run metadata for', epoch)

            # Display logs per epoch step
            if epoch % LOG_FREQUENCY == 0:
                #G_TEST = np.random.uniform(-1., 1., size=[len(TEST_DATASET),GEN_SIZE_IN])
                #G_TEST = np.array(np.random.randint(NUM_CLASSES,size=len(mnist.test.images)),dtype="int32")
                #summary, _g = sess.run([merged, fake_data], feed_dict={x: TEST_DATASET, g: G_TEST})
                #summary,_g = sess.run([merged, fake_data], feed_dict={x: mnist.test.images, g: G_TEST})
                #test_writer.add_summary(summary, i)
                save_path = saver.save(sess, SAVE_PATH, global_step = epoch)
                print('Step %s' % epoch)
        # Cleanup
        train_writer.close()
        test_writer.close()
        print("Training Finished!")

        # Save model weights to disk
        save_path = saver.save(sess, SAVE_PATH)
        print("Model saved in file: %s" % save_path)
