import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(dir_path,"saved")
pickle_file = os.path.join(dir_path,'data','notMNIST.pkl')

with open(pickle_file, 'rb') as f:
    save = pickle.load(f,encoding='latin1')
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)


image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


batch_size = 128
hidden_layer_size = 1024

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_layer_size]))
    biases1 = tf.Variable(tf.zeros([hidden_layer_size]))
    weights2 = tf.Variable(tf.truncated_normal([hidden_layer_size, num_labels]))
    biases2 = tf.Variable(tf.zeros([num_labels]))

    # Training computation. WITH DROPOUT
    keep_prob = tf.placeholder(tf.float32)
    vis_layer_1 = tf.matmul(tf_train_dataset, weights1) + biases1
    dropout_layer_1 = tf.nn.dropout(tf.identity(vis_layer_1),keep_prob)
    hidden_layer_1 = tf.nn.relu(dropout_layer_1)
    dropout_layer_2 = tf.nn.dropout(tf.identity(hidden_layer_1),keep_prob)
    logits = tf.matmul(tf.nn.dropout(dropout_layer_2,keep_prob), weights2) + biases2

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(biases1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(biases2))
    # Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer.
    global_step = tf.Variable(0)  # count the number of steps taken.
    starter_learning_rate = 0.5
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,100000, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset,weights1) + biases1), weights2) + biases2)
    test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset,weights1) + biases1), weights2) + biases2)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

num_steps = 3001

with tf.Session(graph=graph) as session:
    #new_saver = tf.train.import_meta_graph(os.path.join(save_path,'graph.meta'))
    # Restore from path
    #new_saver.restore(session, os.path.join(save_path,"graph"))

    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 0.5}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

    # Define saver
    saver = tf.train.Saver(tf.global_variables())
    saver.save(session, os.path.join(save_path,"graph"))
