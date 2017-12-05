
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import os
dir_path = os.path.dirname(os.path.realpath(__file__))


# Parameters
learning_rate = 0.001
batch_size = 100
max_epochs = 100
display_step = 1
model_path = os.path.join("saved","model.ckpt")
log_dir = "/tmp/tensorflow/log"

# Cleanup previous training log
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MakeDirs(log_dir)

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
img_size = 28
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

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

graph = tf.Graph()
with graph.as_default():
    # tf Graph input

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, n_input], name='x-input')
        y = tf.placeholder(tf.float32, [None, n_classes], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, img_size, img_size, 1])
        tf.summary.image('input', image_shaped_input, n_classes)

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
    hidden1 = layer(x, n_input, n_hidden_1, 'layer1')
    hidden2 = layer(hidden1, n_hidden_1, n_hidden_2, 'layer2')
    pred    = layer(hidden2, n_hidden_2, n_classes, 'out_layer',tf.identity)

    # Define loss function
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Define optimizer
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # Define and track accuracy
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(log_dir,'train'), graph)
    test_writer = tf.summary.FileWriter(os.path.join(log_dir,'test'))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()

#Running first session
with tf.Session(graph=graph) as sess:
    # Initialize variables
    sess.run(init)

    try:
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)
    except Exception as e:
        print("Model restore failed {}".format(e))

    # Training cycle
    for epoch in range(max_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
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
        if epoch % display_step == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (epoch, acc))
    # Cleanup
    train_writer.close()
    test_writer.close()
    print("Training Finished!")

    # Test model
    _,acc = sess.run([correct_prediction,accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("Accuracy:", acc)

    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
