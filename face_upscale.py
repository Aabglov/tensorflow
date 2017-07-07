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
GEN_LEARNING_RATE = 0.0002
ADAM_BETA = 0.5
ORIG_IMG_SIZE1 = 218
ORIG_IMG_SIZE2 = 178

# Resize the images so it doesn't crash my computer
BIG_SIZE1 = int(192 / 2) #64
BIG_SIZE2 = int(160 / 2) #48

IMG_SIZE1 = int(BIG_SIZE1 / 2)
IMG_SIZE2 = int(BIG_SIZE2 / 2)

# SCALE UP
GEN_SIZE_1 = 256 #512 # 1st layer number of features
GEN_SIZE_2 = 64 #256 # 2nd layer number of features
GEN_KERNEL = [3,3]
GEN_STRIDES = (2,2)
SUB_PIXEL = 2

#GEN_SIZE_3 = 32 #128 # 3rd layer
#GEN_SIZE_4 = 16 #64# final layer
#GEN_KERNEL = [5,5]
#DECONV_STRIDES = (2,2)
#CONV_KERNEL = [2,2]

BATCH_SIZE = 50
MAX_STEPS = 40000
LOG_FREQUENCY = 100
NUM_SUMMARY = 4

if tf.gfile.Exists(LOG_DIR):
    tf.gfile.DeleteRecursively(LOG_DIR)
tf.gfile.MakeDirs(LOG_DIR)

######################################### UTILITY FUNCTIONS ########################################
with tf.device(DEVICE):

    # DEFINE MODEL
    graph = tf.Graph()
    with graph.as_default():
        # tf Graph input
        # Make a queue of file names including all the JPEG images files in the relative
        # image directory.
        filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(os.path.join(DATA_PATH,"*.jpg")))

        # Read an entire image file which is required since they're JPEGs, if the images
        # are too large they could be split in advance to smaller files or use the Fixed
        # reader to split up the file.
        image_reader = tf.WholeFileReader()


        # Read a whole file from the queue
        filename, image_file = image_reader.read(filename_queue)

        # Decode the image as a JPEG file, this will turn it into a Tensor which we can
        # then use in training.
        image_orig = tf.image.decode_jpeg(image_file)
        #image_std = tf.image.per_image_standardization(image_orig)
        #image = tf.image.resize_images(image_std, [ORIG_IMG_SIZE1, ORIG_IMG_SIZE2])
        image = tf.image.resize_images(image_orig, [ORIG_IMG_SIZE1, ORIG_IMG_SIZE2])
        image.set_shape((ORIG_IMG_SIZE1, ORIG_IMG_SIZE2, NUM_CHANNELS))

        # Generate batch
        NUM_PROCESS_THREADS = 1
        MIN_QUEUE_EXAMPLES = 100
        images = tf.train.shuffle_batch(
            [image],
            batch_size=BATCH_SIZE,
            num_threads=NUM_PROCESS_THREADS,
            capacity=MIN_QUEUE_EXAMPLES + NUM_CHANNELS * BATCH_SIZE,
            min_after_dequeue=MIN_QUEUE_EXAMPLES)

        # Input placeholders
        with tf.name_scope('input'):
            #x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE], name='x-input')
            img_scaled = (images/127.5) - 1.0
            y = tf.image.resize_images(img_scaled,[BIG_SIZE1,BIG_SIZE2])#images
            x = tf.image.resize_images(img_scaled,[IMG_SIZE1,IMG_SIZE2])#

        with tf.name_scope('input_reshape'):
            image_shaped_true = tf.reshape(y,[-1,BIG_SIZE1, BIG_SIZE2, NUM_CHANNELS])
            tf.summary.image('TRUE', image_shaped_true, NUM_SUMMARY)
            image_shaped_small = tf.reshape(x,[-1,IMG_SIZE1, IMG_SIZE2, NUM_CHANNELS])
            tf.summary.image('INPUT', image_shaped_small, NUM_SUMMARY)
            #shaped_labels = tf.reshape(tf.one_hot(y,NUM_CLASSES),[-1,NUM_CLASSES])

        # Ripped straight from TensorLayers definition
        #   Apply a small negative gradient in relu
        #   instead of 0 for x<=0
        def leaky_relu(x, name, alpha=0.1):
            with tf.name_scope(name) as scope:
                x = tf.maximum(x, alpha * x)
            return x

        def convLayer(input_tensor, kernel_shape, channel_dim, strides, layer_name, act=leaky_relu):
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

        def _phase_shift(I, r):
            # Helper function with main phase shift operation
            bsize, a, b, c = I.get_shape().as_list()
            X = tf.reshape(I, (bsize, a, b, r, r))
            X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
            X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
            X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
            X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
            X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
            return tf.reshape(X, (bsize, a*r, b*r, 1))

        def subPixelLayer(input_tensor, r_size):
            Xc = tf.split(input_tensor, 3, 3)
            X = tf.concat([_phase_shift(x, r_size) for x in Xc], 3)
            return X

        # DEFINE GENERATOR USING DECONVOLUTION
        def generatorDeconv(gen_in):
            # CONVOLVE INPUT TO SMALL KERNEL
            conv1 = convLayer(input_tensor=gen_in, kernel_shape=GEN_KERNEL, channel_dim=NUM_CHANNELS*(SUB_PIXEL**2), strides=GEN_STRIDES, layer_name='gen_conv1')
            subp1 = subPixelLayer(input_tensor=conv1, r_size=SUB_PIXEL)
            conv2 = convLayer(input_tensor=subp1,  kernel_shape=GEN_KERNEL, channel_dim=NUM_CHANNELS*(SUB_PIXEL**2), strides=GEN_STRIDES, layer_name='gen_conv2')
            subp2 = subPixelLayer(input_tensor=conv2, r_size=SUB_PIXEL)
            conv3 = convLayer(input_tensor=subp2,  kernel_shape=GEN_KERNEL, channel_dim=NUM_CHANNELS*(SUB_PIXEL**2), strides=GEN_STRIDES, layer_name='gen_conv3')
            subp3 = subPixelLayer(input_tensor=conv3, r_size=SUB_PIXEL)
            # Don't apply batch normalization to the output layer
            #   Also only use stride of (1,1) so as to make the final subpixel layer increase the image size
            #   Use tanh as the activation based on various papers I've read
            conv4 = tf.layers.conv2d(subp3,NUM_CHANNELS*(SUB_PIXEL**2),GEN_KERNEL,(1,1),padding='same',activation=tf.nn.tanh)
            final = subPixelLayer(input_tensor=conv4, r_size=SUB_PIXEL)
            #deconv2 = deconvLayer(input_tensor=conv1,    channels=GEN_SIZE_3,deconv_kernel=GEN_KERNEL,deconv_strides=DECONV_STRIDES,layer_name="deconv2")
            # Don't apply normalization (batch)
            # to output layer
            #final = tf.layers.conv2d_transpose(inputs=conv1,filters=NUM_CHANNELS,kernel_size=GEN_KERNEL,strides=DECONV_STRIDES,padding='same',activation=tf.nn.tanh)
            image_shaped_gen= tf.reshape(final,[-1, BIG_SIZE1, BIG_SIZE2, NUM_CHANNELS])
            tf.summary.image('generated_input', image_shaped_gen, NUM_SUMMARY)
            return image_shaped_gen


        with tf.variable_scope("generator") as scope:
            fake_data = generatorDeconv(x)
            #fake_data = generator(g)

        with tf.variable_scope("discriminator") as scope:
            diff = tf.squared_difference(image_shaped_true, fake_data)
            #diff = tf.abs(image_shaped_true - fake_data)

        # Define loss function(s)
        with tf.name_scope('loss'):
            generator_loss = tf.reduce_mean(diff)
            tf.summary.scalar('loss', generator_loss)

        # Define optimizer
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(GEN_LEARNING_RATE,beta1=ADAM_BETA).minimize(generator_loss,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'))

        # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph)
        #test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'test'))

        # Initializing the variables
        init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()

    #Running first session
    #def main():
    with tf.Session(graph=graph) as sess:#,config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Initialize variables
        sess.run(init)
        sess.run(local_init)

        try:
            ckpt = tf.train.get_checkpoint_state(CHKPT_PATH)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored from file: %s" % SAVE_PATH)
        except Exception as e:
            print("Model restore failed {}".format(e))

        #train_batcher = Batcher(TRAIN_DATASET,TRAIN_LABELS)
        #total_batch = int(len(TRAIN_DATASET)/BATCH_SIZE)
        #Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Training cycle
        for epoch in range(1,MAX_STEPS):

            # Run optimization op (backprop) and cost op (to get loss value)
            summary,img, g_unused, _g = sess.run([merged, image, fake_data, train_step], feed_dict={})
            train_writer.add_summary(summary, epoch)
            print('Adding run data for', epoch)


            # Display logs per epoch step
            if epoch % LOG_FREQUENCY == 0:
                #   I'm not a billion percent sure what this does....
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, img, _g = sess.run([merged, image, train_step],
                                      feed_dict={},
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, "step_{}".format(epoch))
                train_writer.add_summary(summary, epoch)
                print('Adding run metadata for', epoch)
                save_path = saver.save(sess, SAVE_PATH, global_step = epoch)
                print('Step %s' % epoch)
        # Cleanup
        #   Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)
        #   Close writers
        train_writer.close()
        #test_writer.close()
        print("Training Finished!")

        # Save model weights to disk
        save_path = saver.save(sess, SAVE_PATH)
        print("Model saved in file: %s" % save_path)
