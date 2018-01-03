# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import tensorflow as tf
import pickle

# PATHS -- absolute
dir_path = os.path.dirname(os.path.realpath(__file__))
checkpoint_path = os.path.join(dir_path,"saved","no_back")#,"mtg_rec_char_steps.ckpt")

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

    x = np.array(x)
    y = np.array(y)

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

num_examples = 1000
output_dim = 12
iterations = 100

x,y = generate_dataset(num_examples=num_examples, output_dim = output_dim)

batch_size = 100#0
alpha = 0.01#0.0001

input_dim = len(x[0])
layer_1_dim = 128
layer_2_dim = 64
output_dim = len(y[0])

graph = tf.Graph()
with graph.as_default():

    # Placeholders
    with tf.name_scope("input"):
        x = tf.placeholder(tf.int32, [None, input_dim], name='x')
        y = tf.placeholder(tf.int32, [None, output_dim], name='y')

    # Get dynamic batch_size
    # Not sure this is actually needed
    batch_size = tf.shape(x)[0]

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
                with tf.name_scope('synthetic_gradient'):
                    with tf.name_scope('weights'):
                        self.var_weights_synthetic_grads = init_weight([output_dim, output_dim])
                        if self.summarize:
                            variable_summaries(self.var_weights_synthetic_grads)
                    with tf.name_scope('biases'):
                        self.var_bias_synthetic_grads = init_bias([output_dim])
                        if self.summarize:
                            variable_summaries(self.var_bias_synthetic_grads)
                with tf.name_scope('activation'):
                    self.act = act

        def forward_and_synthetic_update(self,input_tensor):
            with tf.name_scope(self.name):
                with tf.name_scope('preactivation'):
                    self.preactivate = tf.matmul(input_tensor, self.weights) + self.biases
                    if self.summarize:
                        tf.summary.histogram('pre_activations', self.preactivate)
                with tf.name_scope('activation'):
                    self.activations = self.act(self.preactivate, name='activation')
                    if self.summarize:
                        tf.summary.histogram('activations', self.activations)

                with tf.name_scope('synthetic_gradient'):
                    self.synthetic_gradient = (self.activations.dot(self.var_weights_synthetic_grads) + self.var_bias_synthetic_grads)
                    self.weight_synthetic_gradient = self.synthetic_gradient * tf.gradients(self.activations,self.pre_activations)
                    self.synthetic_gradient_output = self.weight_synthetic_gradient.dot(self.weights.T)
                    if self.summarize:
                        variable_summaries(self.synthetic_gradient)
                        variable_summaries(self.weight_synthetic_gradient)
                        variable_summaries(self.synthetic_gradient_output)

            return self.synthetic_gradient_output,self.activations

        def update_synthetic_weights(self,true_gradient):
            with tf.name_scope(self.name):
                with tf.name_scope('update'):
                    with tf.name_scope('synthetic_gradient'):
                        self.synthetic_gradient_delta = (self.synthetic_gradient - true_gradient)
                        self.var_weights_synthetic_grads -= self.output.T.dot(self.synthetic_gradient_delta) * self.alpha
                        self.var_bias_synthetic_grads -= np.average(self.synthetic_gradient_delta,axis=0) * self.alpha


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
