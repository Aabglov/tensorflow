import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import time

# Set random seed
seed = 36 # Pick your favorite
np.random.seed(seed)
tf.set_random_seed(seed)

######################################### CONSTANTS ########################################
NUM_CLASSES = 10 # The number of digits present in the dataset
DEVICE = "/cpu:0" # Controls whether we run on CPU or GPU
INPUT_SIZE = 28 # Size of an image in dataset
NUM_CHANNELS = 1

FLAGS = tf.app.flags.FLAGS

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
TRAIN_DIR = "/tmp/tensorflow/log"
DATA_PATH = os.path.join(DIR_PATH,"data","notMNIST.pkl")
SAVE_DIR = os.path.join(DIR_PATH,"saved","conv")
BATCH_SIZE = 128
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
MAX_STEPS = 1000000
LOG_FREQUENCY = 10

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1

######################################### UTILITY FUNCTIONS ########################################
def genRandNum():
    return np.random.randint(NUM_CLASSES)

# Dynamic One-Hot Encoder
#   Works on int, list and Numpy Arrays
def onehot(x):
    if type(x).__name__ in ["int","float"]:
        oh = np.zeros((1,NUM_CLASSES))
        oh[0,x] = 1.0
        return oh
    elif type(x).__name__ in ["ndarray","list"]:
        if type(x).__name__ == "list":
            num = len(x)
        elif type(x).__name__ == "ndarray":
            num = x.shape[0]
        oh = np.zeros((num,NUM_CLASSES))
        oh[np.arange(num), x] = 1.0
        return oh


######################################## SUMMARY HELPERS #################################
def activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = x.op.name#re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


######################################### MODEL ########################################
def model(images):
    # CONV NET
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable('weights',shape=[5, 5, NUM_CHANNELS, 64],initializer=tf.truncated_normal_initializer(stddev=5e-2))
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable('weights', shape=[5, 5, 64, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2))
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights', shape=[dim, 384], initializer=tf.truncated_normal_initializer(stddev=0.04))
        # Add weight decay for this layer
        weight_decay = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        #
        biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        activation_summary(local3)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights', shape=[384, NUM_CLASSES], initializer=tf.truncated_normal_initializer(stddev=1/384.0))
        biases = tf.get_variable('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local3, weights), biases, name=scope.name)
        activation_summary(softmax_linear)

    return softmax_linear

def calculateLoss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    # NOTE: We should only have one other value in this collection (local3)
    # but that may change
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def add_loss_summaries(total_loss):
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


# TODO:
# FIX EVERYTHING BELOW THIS LINE
# Test model, try different structure
# Add notMNIST data in place of cifar

def train(total_loss, global_step):
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op




def main(argv=None):
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        ######################################### LOAD DATA ########################################
        with open(DATA_PATH,"rb") as f:
            data = pickle.load(f)
        # Image dataset needs to be shaped into 4D tensor to accomodate convolution operator.
        # If these images were color this new dimension would be the color channels.
        #   [num_examples, in_height, in_width, in_channels]
        train_dataset = data['train_dataset'].reshape((-1,INPUT_SIZE,INPUT_SIZE,1))
        train_labels = data['train_labels']
        test_dataset = data['test_dataset'].reshape((-1,INPUT_SIZE,INPUT_SIZE,1))
        test_labels = data['test_labels']
        valid_dataset = data['valid_dataset'].reshape((-1,INPUT_SIZE,INPUT_SIZE,1))
        valid_labels = data['valid_labels']

        # Generate a batch queue
        train_batch, label_batch = tf.train.batch([train_dataset, train_labels],batch_size=BATCH_SIZE,capacity=30,enqueue_many=True)

        tf.summary.image('images', train_batch)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model(train_batch)

        # Calculate loss.
        loss = calculateLoss(logits, label_batch)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries)
        summary_writer = tf.summary.FileWriter(TRAIN_DIR)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % LOG_FREQUENCY == 0:

                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = LOG_FREQUENCY * BATCH_SIZE / duration
                    sec_per_batch = float(duration / LOG_FREQUENCY)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=SAVE_DIR,
                hooks=[tf.train.StopAtStepHook(last_step=MAX_STEPS),tf.train.NanTensorHook(loss),_LoggerHook()],
                config=tf.ConfigProto(log_device_placement=False)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


if __name__ == '__main__':
    tf.app.run()
