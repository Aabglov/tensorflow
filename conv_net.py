import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import time

# PATHS
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = os.path.join(DIR_PATH,"saved","conv","model.ckpt")
LOG_DIR = "/tmp/tensorflow/log"#os.path.join(DIR_PATH,"tensorboard","conv")
DATA_PATH = os.path.join(DIR_PATH,"data","notMNIST.pkl")
CSV_PATH = os.path.join(DIR_PATH,"test.csv")


# with open(CSV_PATH,"w+") as f:
#     for i in range(len(train)):
#         f.write(",".join(["{:f}".format(t) for t in train[i].ravel()]) +
#                 "," +
#                 "{:f}".format(labels[i]) +
#                 "\n")
#
# HODOR

# Set random seed
seed = 36 # Pick your favorite
np.random.seed(seed)
tf.set_random_seed(seed)

######################################### CONSTANTS ########################################
NUM_CLASSES = 10 # The number of digits present in the dataset
DEVICE = "/cpu:0" # Controls whether we run on CPU or GPU
IMG_SIZE = 28 # Size of an image in dataset
NUM_CHANNELS = 1
LEARNING_RATE = 0.001

HIDDEN_SIZE_1 = 256 # 1st layer number of features
HIDDEN_SIZE_2 = 256 # 2nd layer number of features
INPUT_SIZE = 784 # MNIST data input (img shape: 28*28)
NUM_CLASSES = 10 # MNIST total classes (0-9 digits)

BATCH_SIZE = 128
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
MAX_STEPS = 10000
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


# DEFINE MODEL
graph = tf.Graph()
with graph.as_default():
    # tf Graph input

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE], name='x-input')
        y = tf.placeholder(tf.int32, [None,], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x,[-1,IMG_SIZE, IMG_SIZE, 1])
        tf.summary.image('input', image_shaped_input, NUM_CLASSES)
        shaped_input = tf.reshape(x, [-1, INPUT_SIZE])
        shaped_labels = tf.one_hot(y,NUM_CLASSES)

    # We can't initialize these variables to 0 - the network will get stuck.
    def init_weight(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        #return tf.get_variable(shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
        return tf.Variable(initial)

    def init_bias(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        # return  tf.get_variable(shape=shape, initializer=tf.constant_initializer(0.1))
        return tf.Variable(initial)

    # Create model
    def layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = init_weight([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = init_bias([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    # Create our 3-layer model
    hidden1 = layer(shaped_input, INPUT_SIZE, HIDDEN_SIZE_1, 'layer1')
    hidden2 = layer(hidden1, HIDDEN_SIZE_1, HIDDEN_SIZE_2, 'layer2')
    pred =    layer(hidden2, HIDDEN_SIZE_2, NUM_CLASSES, 'out_layer',tf.identity)

    # Define loss function
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=shaped_labels, logits=pred)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Define optimizer
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    # Define and track accuracy
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(shaped_labels, 1))
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
def main():
    with tf.Session(graph=graph) as sess:
        # Initialize variables
        sess.run(init)

        try:
            saver.restore(sess, SAVE_PATH)
            print("Model restored from file: %s" % SAVE_PATH)
        except Exception as e:
            print("Model restore failed {}".format(e))

        train_batcher = Batcher(TRAIN_DATASET,TRAIN_LABELS)
        #test_batcher =  Batcher(TEST_DATASET, TEST_DATASET)
        total_batch = int(len(TRAIN_DATASET)/BATCH_SIZE)

        # Training cycle
        for epoch in range(MAX_STEPS):
            avg_cost = 0.

            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = train_batcher.nextBatch(BATCH_SIZE)
                # Run optimization op (backprop) and cost op (to get loss value)
                summary, _ = sess.run([merged, train_step], feed_dict={x: batch_x, y: batch_y})
                train_writer.add_summary(summary, i)

                # Keep track of meta data
                if i % 100 == 0:
                    #   I'm not a billion percent sure what this does....
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step],
                                          feed_dict={x: batch_x, y: batch_y},
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, "step_{}_{}".format(epoch,i))
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)

            # Display logs per epoch step
            if epoch % LOG_FREQUENCY == 0:
                summary, acc = sess.run([merged, accuracy], feed_dict={x: TEST_DATASET, y: TEST_LABELS})
                test_writer.add_summary(summary, i)
                print('Accuracy at step %s: %s' % (epoch, acc))
        # Cleanup
        train_writer.close()
        test_writer.close()
        print("Training Finished!")

        # Test model
        _,acc = sess.run([correct_prediction,accuracy], feed_dict={x: TEST_DATASET, y: TEST_LABELS})
        print("Accuracy:", acc)

        # Save model weights to disk
        save_path = saver.save(sess, SAVE_PATH)
        print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    # Cleanup previous training log
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
    main()
