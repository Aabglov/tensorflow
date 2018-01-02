# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import tensorflow as tf
import os
import word_helpers
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
    x = tf.placeholder(tf.int32, [None, input_dim], name='input_placeholder')
    y = tf.placeholder(tf.int32, [None, output_dim], name='labels_placeholder')

    # Get dynamic batch_size
    batch_size = tf.shape(x)[0]


     class DNI(object):

         def __init__(self,input_dim, output_dim,nonlin,nonlin_deriv,alpha = 0.1):

             self.weights = (np.random.randn(input_dim, output_dim) * 2) - 1
             self.bias = (np.random.randn(output_dim) * 2) - 1

             self.weights_0_1_synthetic_grads = (np.random.randn(output_dim,output_dim) * .0) - .0
             self.bias_0_1_synthetic_grads = (np.random.randn(output_dim) * .0) - .0

             self.nonlin = nonlin
             self.nonlin_deriv = nonlin_deriv
             self.alpha = alpha

         def forward_and_synthetic_update(self,input):

             self.input = input
             self.output = self.nonlin(self.input.dot(self.weights)  + self.bias)

             self.synthetic_gradient = (self.output.dot(self.weights_0_1_synthetic_grads) + self.bias_0_1_synthetic_grads)
             self.weight_synthetic_gradient = self.synthetic_gradient * self.nonlin_deriv(self.output)

             self.weights -= self.input.T.dot(self.weight_synthetic_gradient) * self.alpha
             self.bias -= np.average(self.weight_synthetic_gradient,axis=0) * self.alpha
             
             return self.weight_synthetic_gradient.dot(self.weights.T), self.output

         def update_synthetic_weights(self,true_gradient):
             self.synthetic_gradient_delta = (self.synthetic_gradient - true_gradient)
             self.weights_0_1_synthetic_grads -= self.output.T.dot(self.synthetic_gradient_delta) * self.alpha
             self.bias_0_1_synthetic_grads -= np.average(self.synthetic_gradient_delta,axis=0) * self.alpha

    def convLayer(input_tensor, kernel_shape, channel_dim, strides, layer_name, dr=0.2, pool_size=3, act=leaky_relu):
        with tf.variable_scope(layer_name) as scope:
            # 2D Convolution
            conv = tf.layers.conv2d(input_tensor,channel_dim,kernel_shape,strides=strides,padding='same',activation=None)
            # Pooling
            #pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[pool_size,pool_size], strides=pool_size)
            # Psuedo down-sampling
            #down = tf.layers.conv2d(conv, channel_dim, [pool_size,pool_size], (pool_size,pool_size), padding='valid', activation=None)
            #norm = tf.layers.dropout(inputs=act_out, rate=dr)
            norm = act(tf.layers.batch_normalization(conv,momentum=0.9,epsilon=1e-5,training=True),name=layer_name, alpha=0.1)
            return norm

    def deconvLayer(input_tensor, channels, deconv_kernel, deconv_strides, layer_name, conv_kernel=[3,3], conv_strides=(1,1), act=tf.nn.relu):
        with tf.variable_scope(layer_name) as scope:
            #conv =   tf.layers.conv2d(inputs=input_tensor,filters=channels,kernel_size=conv_kernel,strides=conv_strides,padding='same',activation=None)
            deconv = tf.layers.conv2d_transpose(inputs=input_tensor,filters=channels,kernel_size=deconv_kernel,strides=deconv_strides,padding='same',activation=None)
            norm = act(tf.layers.batch_normalization(deconv,momentum=0.9,epsilon=1e-5,training=True))
            return norm

    # DEFINE GENERATOR USING DECONVOLUTION
    def generatorDeconv(gen_in):
        linear = tf.layers.dense(inputs=gen_in, units=GEN_IN_X*GEN_IN_Y*GEN_CHANNELS, activation=tf.nn.relu)
        shaped_in = tf.reshape(linear,[-1,GEN_IN_X,GEN_IN_Y,GEN_CHANNELS])
        deconv1 = deconvLayer(input_tensor=shaped_in ,channels=GEN_SIZE_1,deconv_kernel=GEN_KERNEL,deconv_strides=GEN_STRIDES,layer_name="deconv1")
        deconv2 = deconvLayer(input_tensor=deconv1,channels=GEN_SIZE_2,deconv_kernel=GEN_KERNEL,deconv_strides=GEN_STRIDES,layer_name="deconv2")
        deconv3 = deconvLayer(input_tensor=deconv2,channels=GEN_SIZE_3,deconv_kernel=GEN_KERNEL,deconv_strides=GEN_STRIDES,layer_name="deconv3")
        #deconv4 = deconvLayer(input_tensor=deconv3,channels=GEN_SIZE_4,deconv_kernel=GEN_KERNEL,deconv_strides=(2,2),conv_kernel=CONV_KERNEL,conv_strides=(2,2),layer_name="deconv4")
        deconv_out = tf.layers.conv2d_transpose(inputs=deconv3,filters=NUM_CHANNELS,kernel_size=GEN_KERNEL,strides=GEN_STRIDES,padding='same',activation=tf.nn.tanh)
        #flat = tf.contrib.layers.flatten(deconv_out)
        #dense = tf.layers.dense(inputs=flat, units=IMG_SIZE1*IMG_SIZE2*NUM_CHANNELS, activation=tf.nn.relu)
        image_shaped_gen= tf.reshape(deconv_out,[-1,IMG_SIZE1, IMG_SIZE2, NUM_CHANNELS])
        tf.summary.image('generated_input', image_shaped_gen, 4)
        #return gen2
        return image_shaped_gen

    # DEFINE DISCRIMINATOR
    def discriminatorConv(input_tensor):
        #hidden1 =    convLayer(input_tensor, DISC_KERNEL,  HIDDEN_SIZE_1, 'layer1')
        # Don't apply batch normalization to input layer
        with tf.variable_scope("layer1") as scope:
            hidden1 = tf.layers.conv2d(input_tensor,HIDDEN_SIZE_1,DISC_KERNEL,strides=(2,2),padding='same',activation=tf.nn.relu)
        hidden2 =    convLayer(hidden1,      DISC_KERNEL,  HIDDEN_SIZE_2, (2,2), 'layer2')
        hidden3 =    convLayer(hidden2,      DISC_KERNEL,  HIDDEN_SIZE_3, (2,2), 'layer3')
        hidden_out = convLayer(hidden3,      DISC_KERNEL,  HIDDEN_SIZE_4, (2,2), 'layer_out')
        # Dense Layer
        with tf.variable_scope("dense") as scope:
            flat = tf.contrib.layers.flatten(hidden_out)
            #dense = tf.layers.dense(inputs=flat, units=HIDDEN_SIZE_4, activation=tf.nn.relu)
            # Logits Layer
            #dropout = tf.layers.dropout(inputs=dense, rate=0.2)
            logits = tf.layers.dense(inputs=flat, units=1)
        prob = tf.nn.sigmoid(logits)
        return prob, logits


    with tf.variable_scope("generator") as scope:
        fake_data = generatorDeconv(g_shaped)

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
