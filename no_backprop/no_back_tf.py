# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import tensorflow as tf
import pickle
import time

# PATHS -- absolute
dir_path = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = "no_back"
checkpoint_path = os.path.join(dir_path,SAVE_DIR)#,"mtg_rec_char_steps.ckpt")
CHECKPOINT_NAME = "dict_steps.ckpt"
LOG_DIR = os.path.join(dir_path,SAVE_DIR,"log")
model_path = os.path.join(dir_path,SAVE_DIR,CHECKPOINT_NAME)

num_examples = 10000
NUM_EPOCHS = 10000
LOG_EPOCH = 100

batch_size = 1000
alpha = 0.01#0.0001
LEARNING_RATE = 0.001
ADAM_BETA = 0.5

layer_1_dim = 128
layer_2_dim = 64
output_dim = 12
input_dim = int(2 * output_dim)

def generate_dataset(output_dim = 8,num_examples=1000):
    def int2vec(x,dim=output_dim):
        out = np.zeros(dim)
        binrep = np.array(list(np.binary_repr(x))).astype('int')
        out[-len(binrep):] = binrep
        return out

    x_left_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    x_right_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    y_int = x_left_int + x_right_int

    x = list()
    for i in range(len(x_left_int)):
        x.append(np.concatenate((int2vec(x_left_int[i]),int2vec(x_right_int[i]))))

    y = list()
    for i in range(len(y_int)):
        y.append(int2vec(y_int[i]))

    x = np.array(x,dtype="float32")
    y = np.array(y,dtype="float32")

    return (x,y)

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

    # Placeholders
    with tf.name_scope("input"):
        x_input = tf.placeholder(shape=[None, input_dim], dtype=tf.float32, name='x_input')
        y_input = tf.placeholder(shape=[None, output_dim],dtype=tf.float32, name='y_input')

    # Get dynamic batch_size
    # Not sure this is actually needed
    #batch_size = tf.shape(x)[0]

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


    class DNI:
        def __init__(self,input_dim, output_dim, layer_name, act=tf.sigmoid, alpha=0.1, summarize=False):
            self.name = layer_name
            self.summarize = summarize
            self.alpha = alpha
            with tf.name_scope(self.name):
                with tf.name_scope('forward'):
                    # This Variable will hold the state of the weights for the layer
                    with tf.name_scope('weights'):
                        self.weights = init_weight([input_dim, output_dim])
                        if self.summarize:
                            variable_summaries(self.weights)
                    with tf.name_scope('biases'):
                        self.biases = init_bias([output_dim])
                        if self.summarize:
                            variable_summaries(self.biases)
                with tf.name_scope('TRAINING_VARIABLES'):
                    with tf.name_scope('weights'):
                        self.var_weights_synthetic_grads = init_weight([output_dim, output_dim])
                        if self.summarize:
                            variable_summaries(self.var_weights_synthetic_grads)
                    with tf.name_scope('biases'):
                        self.var_bias_synthetic_grads = init_bias([output_dim])
                        if self.summarize:
                            variable_summaries(self.var_bias_synthetic_grads)
                with tf.name_scope('activation_function'):
                    self.act = act

        def forward_and_synthetic_update(self,input_tensor):
            with tf.name_scope(self.name):
                with tf.name_scope('pre_activation'):
                    self.pre_activations = tf.matmul(input_tensor, self.weights) + self.biases
                    if self.summarize:
                        tf.summary.histogram('pre_activations', self.pre_activations)
                with tf.name_scope('activation'):
                    self.activations = self.act(self.pre_activations)
                    if self.summarize:
                        tf.summary.histogram('activations', self.activations)

                with tf.name_scope('synthetic_gradient'):
                    self.synthetic_gradient = tf.matmul(self.activations, self.var_weights_synthetic_grads) + self.var_bias_synthetic_grads
                    self.weight_synthetic_gradient = self.synthetic_gradient * tf.gradients(self.activations,self.pre_activations)[0]
                    self.synthetic_gradient_output = tf.matmul(self.weight_synthetic_gradient,tf.transpose(self.weights))
                    if self.summarize:
                        variable_summaries(self.synthetic_gradient)
                        variable_summaries(self.weight_synthetic_gradient)
                        variable_summaries(self.synthetic_gradient_output)

            return self.synthetic_gradient_output,self.activations


    class NormalLayer:
        def __init__(self,input_dim, output_dim, layer_name, act=tf.sigmoid, alpha=0.1, summarize=False):
            self.name = layer_name
            self.summarize = summarize
            self.alpha = alpha
            with tf.name_scope(self.name):
                with tf.name_scope('TRAINING_VARIABLES'):
                    # This Variable will hold the state of the weights for the layer
                    with tf.name_scope('weights'):
                        self.weights = init_weight([input_dim, output_dim])
                        if self.summarize:
                            variable_summaries(self.weights)
                    with tf.name_scope('biases'):
                        self.biases = init_bias([output_dim])
                        if self.summarize:
                            variable_summaries(self.biases)
                with tf.name_scope('activation_function'):
                    self.act = act

        def forward(self,input_tensor):
            with tf.name_scope(self.name):
                with tf.name_scope('pre_activation'):
                    self.pre_activations = tf.matmul(input_tensor, self.weights) + self.biases
                    if self.summarize:
                        tf.summary.histogram('pre_activations', self.pre_activations)
                with tf.name_scope('activation'):
                    self.activations = self.act(self.pre_activations)
                    if self.summarize:
                        tf.summary.histogram('activations', self.activations)

            return self.activations

    # Model
    with tf.name_scope("model"):
        layer_1 = DNI(input_dim, layer_1_dim, "dni_1", act=tf.sigmoid, alpha=0.01, summarize=True)
        layer_2 = DNI(layer_1_dim, layer_2_dim, "dni_2", act=tf.sigmoid, alpha=0.01, summarize=True)
        layer_3 = NormalLayer(layer_2_dim, output_dim, "layer_3", act=tf.sigmoid, alpha=0.01, summarize=True)

    # Forward Pass
    with tf.name_scope("forward"):
        _, layer_1_out = layer_1.forward_and_synthetic_update(x_input)
        layer_1_delta, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out)
        # Normal update
        layer_3_out = layer_3.forward(layer_2_out)

    # Loss
    with tf.name_scope("loss"):
        #layer_3_delta = (layer_3_out - y_input
        #loss = tf.reduce_sum(layer_3_delta)
        loss = tf.losses.mean_squared_error(y_input,layer_3_out)
        tf.summary.scalar('loss', loss)

    # Backward Propagation
    with tf.name_scope('train'):
        #layer_2_delta = layer_3.backward(layer_3_delta)
        #layer_3_updated = layer_3.update(layer_2_out)
        #layer_2_updated = layer_2.update_synthetic_weights(layer_2_delta)
        #layer_1_updated = layer_1.update_synthetic_weights(layer_1_delta)
        # .*\/synthetic\/.*
        collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='.*\/TRAINING_VARIABLES\/.*')#[]
        #for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='.*\/TRAINING_VARIABLES\/.*')
        #    if "synthetic_gradient" in var.name_scope:
        #        print(var)
        #        collection.append(var)
        #print(len(collection))
        train_step = tf.train.AdamOptimizer(LEARNING_RATE,beta1=ADAM_BETA).minimize(loss,var_list=collection)

    # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph)

    # Initializing the variables
    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()

    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()


print("Beginning Session")
x,y = generate_dataset(num_examples=num_examples, output_dim = output_dim)

#Running first session
with tf.Session(graph=graph) as sess:
    # Initialize variables
    #sess.run(init)
    sess.run(tf.global_variables_initializer())

    try:
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored from file: %s" % model_path)
    except Exception as e:
        print("Model restore failed {}".format(e))

    # Training cycle
    already_trained = 0
    for epoch in range(already_trained,NUM_EPOCHS):
        for batch_i in range(int(num_examples / batch_size)):
            batch_x = x[(batch_i * batch_size):(batch_i+1)*batch_size]
            batch_y = y[(batch_i * batch_size):(batch_i+1)*batch_size]

            start = time.time()

            # Run optimization op (backprop) and cost op (to get loss value)
            summary,cost,_ = sess.run([merged,loss,train_step], feed_dict={x_input: batch_x, y_input: batch_y})
            avg_cost = cost/batch_size
            end = time.time()
            train_writer.add_summary(summary, epoch)
            print("Epoch:", '{}'.format(epoch), "cost=" , "{}".format(avg_cost), "time:", "{}".format(end-start))

        # Display logs per epoch step
        if epoch % LOG_EPOCH == 0:
            #train_writer.add_run_metadata(run_metadata, "step_{}".format(epoch))
            print("saving {}".format(epoch)) # Spacer
            save_path = saver.save(sess, model_path, global_step = epoch)

    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
