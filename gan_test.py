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
DIS_LEARNING_RATE = 0.001 #0.001
GEN_LEARNING_RATE = 0.001 #0.001
ADAM_BETA = 0.5
ORIG_IMG_SIZE1 = 218
ORIG_IMG_SIZE2 = 178

# Resize the images so it doesn't crash my computer
IMG_SIZE1 = 64
IMG_SIZE2 = 48

# GENERATOR
GEN_SIZE_IN = 100
GEN_IN_X = 4#20
GEN_IN_Y = 3#17
GEN_CHANNELS = 1024
GEN_TOTAL_IN = GEN_IN_X * GEN_IN_Y * GEN_CHANNELS
GEN_SIZE_1 = 512 # 1st layer number of features
GEN_SIZE_2 = 256 # 2nd layer number of features
GEN_SIZE_3 = 128 # 3rd layer
GEN_SIZE_4 = 64# final layer
GEN_KERNEL = [5,5]
GEN_STRIDES = (2,2)
CONV_KERNEL = [2,2]
SUB_PIXEL = 4

# DISCRIMINATOR
HIDDEN_SIZE_1 = 32 # 1st layer number of features
HIDDEN_SIZE_2 = 64 # 2nd layer number of features
HIDDEN_SIZE_3 = 128 # 3rd layer
HIDDEN_SIZE_4 = 256 # Dense layer -- ouput
DISC_KERNEL = [5,5]

BATCH_SIZE = 25
MAX_STEPS = 10000
LOG_FREQUENCY = 100


################################# SIZE TESTING ####################################################
def init_weight(shape,name,sd=None):
           """Create a weight variable with appropriate initialization."""
           if not sd:
               sd = 1. / tf.sqrt(shape[0] / 2.)
           initial = tf.truncated_normal(shape, stddev=sd)
           return tf.get_variable(name="{}_weights".format(name), initializer=initial)

def convLayer(input_tensor, kernel_shape, channel_dim, strides, layer_name, dr=0.2, pool_size=3, act=tf.nn.relu):
    with tf.variable_scope(layer_name) as scope:
        # 2D Convolution
        conv = tf.layers.conv2d(input_tensor,channel_dim,kernel_shape,strides=strides,padding='same',activation=None)
        # Pooling
        #pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[pool_size,pool_size], strides=pool_size)
        # Psuedo down-sampling
        #down = tf.layers.conv2d(conv, channel_dim, [pool_size,pool_size], (pool_size,pool_size), padding='valid', activation=None)
        #norm = tf.layers.dropout(inputs=act_out, rate=dr)
        norm = act(tf.layers.batch_normalization(conv,momentum=0.9,epsilon=1e-5,training=True))
        return norm


def deconvLayer(input_tensor, channel_dim, kernel_shape, strides, layer_name, act=tf.nn.relu):
    with tf.variable_scope(layer_name) as scope:
        deconv = tf.layers.conv2d_transpose(inputs=input_tensor,filters=channel_dim,kernel_size=kernel_shape,strides=strides,padding='same',activation=None)
        norm = act(tf.layers.batch_normalization(deconv,momentum=0.9,epsilon=1e-5,training=True))
        return norm


#input_tensor = init_weight([1,IMG_SIZE1,IMG_SIZE2,3],"input")
#conv_layer = tf.layers.conv2d(input_tensor, 3, kernel_size=[5,5], strides=(2,2), padding='same',activation=None)

gen_input = init_weight([112,4,3,3],"genput")
#conv1 = convLayer(input_tensor=gen_input, kernel_shape=GEN_KERNEL, channel_dim=GEN_SIZE_1, strides=GEN_STRIDES, layer_name='gen_conv1')
#conv2 = convLayer(input_tensor=conv1,     kernel_shape=GEN_KERNEL, channel_dim=GEN_SIZE_2, strides=GEN_STRIDES, layer_name='gen_conv2')
#conv3 = convLayer(input_tensor=conv2,     kernel_shape=GEN_KERNEL, channel_dim=GEN_SIZE_3, strides=GEN_STRIDES, layer_name='gen_conv3')
#conv4 = convLayer(input_tensor=conv3,     kernel_shape=GEN_KERNEL, channel_dim=GEN_SIZE_4, strides=GEN_STRIDES, layer_name='gen_conv4')

deconv1 = deconvLayer(input_tensor=gen_input, kernel_shape=GEN_KERNEL, channel_dim=GEN_SIZE_4, strides=GEN_STRIDES, layer_name="deconv1")
deconv2 = deconvLayer(input_tensor=deconv1  , kernel_shape=GEN_KERNEL, channel_dim=GEN_SIZE_3, strides=GEN_STRIDES, layer_name="deconv2")
deconv3 = deconvLayer(input_tensor=deconv2  , kernel_shape=GEN_KERNEL, channel_dim=GEN_SIZE_2, strides=GEN_STRIDES, layer_name="deconv3")
deconv4 = deconvLayer(input_tensor=deconv3  , kernel_shape=GEN_KERNEL, channel_dim=GEN_SIZE_1, strides=GEN_STRIDES, layer_name="deconv4")

HODOR
#
# #Make a queue of file names including all the JPEG images files in the relative
# #image directory.
# filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(os.path.join(DATA_PATH,"*.jpg")))
#
# #Read an entire image file which is required since they're JPEGs, if the images
# #are too large they could be split in advance to smaller files or use the Fixed
# #reader to split up the file.
# image_reader = tf.WholeFileReader()
#
# #Read a whole file from the queue
# filename, image_file = image_reader.read(filename_queue)
#
# #Decode the image as a JPEG file, this will turn it into a Tensor which we can
# #then use in training.
# image = tf.image.decode_jpeg(image_file)
#
# with tf.Session() as sess:
#     # Required to get the filename matching to run.
#     tf.global_variables_initializer().run()
#     tf.local_variables_initializer().run()
#     # Coordinate the loading of image files.
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     # Get an image tensor and print its value.
#     for _ in range(1):
#         key,image_tensor = sess.run([filename,image])
#         print(key)
#         print(image_tensor)
