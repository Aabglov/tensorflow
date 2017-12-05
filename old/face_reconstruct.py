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
IMG_SIZE1 = 192 #64
IMG_SIZE2 = 160 #48

BLACKOUT_SIZE1 = int(IMG_SIZE1 / 2)
BLACKOUT_SIZE2 = int(IMG_SIZE2 / 2)
BLACKOUT_INDEX1 = int(IMG_SIZE1 / 4)
BLACKOUT_INDEX2 = int(IMG_SIZE2 / 4)

# SCALE UP
GEN_SIZE_1 = 512 # 1st layer number of features
GEN_SIZE_2 = 256 # 2nd layer number of features
GEN_SIZE_3 = 128 # 3rd layer
GEN_SIZE_4 = 64# final layer
GEN_KERNEL = [5,5]
DECONV_STRIDES = (2,2)
CONV_KERNEL = [2,2]

# SCALE DOWN
HIDDEN_SIZE_1 = 32 # 1st layer number of features
HIDDEN_SIZE_2 = 64 # 2nd layer number of features
HIDDEN_SIZE_3 = 128 # 3rd layer
HIDDEN_SIZE_4 = 256 # 4th layer
HIDDEN_SIZE_5 = 512 # 5th layer
DISC_KERNEL = [5,5]

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

        def blackout(x):
            b = np.ones(x.get_shape())
            #print(b.shape)
            b[:,BLACKOUT_INDEX1:BLACKOUT_INDEX1+BLACKOUT_SIZE1,BLACKOUT_INDEX2:BLACKOUT_INDEX2+BLACKOUT_SIZE2,:] = 0.
            return x * b

        # Input placeholders
        with tf.name_scope('input'):
            #x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE], name='x-input')
            img_scaled = (images/127.5) - 1.0
            y = tf.image.resize_images(img_scaled,[IMG_SIZE1,IMG_SIZE2])#images
            x = blackout(y)

        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(y,[-1,IMG_SIZE1, IMG_SIZE2, NUM_CHANNELS])
            tf.summary.image('TRUE', image_shaped_input, NUM_SUMMARY)
            image_shaped_blackout = tf.reshape(x,[-1,IMG_SIZE1, IMG_SIZE2, NUM_CHANNELS])
            tf.summary.image('BLACKOUT', image_shaped_blackout, NUM_SUMMARY)
            #shaped_labels = tf.reshape(tf.one_hot(y,NUM_CLASSES),[-1,NUM_CLASSES])

        # Ripped straight from TensorLayers definition
        #   Apply a small negative gradient in relu
        #   instead of 0 for x<=0
        def leaky_relu(x, name, alpha=0.1):
            with tf.name_scope(name) as scope:
                x = tf.maximum(x, alpha * x)
            return x

        def convLayer(input_tensor, kernel_shape, channel_dim, layer_name, dr=0.2, pool_size=3, act=leaky_relu):
            with tf.variable_scope(layer_name) as scope:
                # 2D Convolution
                conv = tf.layers.conv2d(input_tensor,channel_dim,kernel_shape,strides=(2,2),padding='same',activation=None)
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
            # CONVOLVE INPUT TO SMALL KERNEL
            hidden1 =    convLayer(gen_in,       DISC_KERNEL,  HIDDEN_SIZE_1, 'gen_conv1')
            hidden2 =    convLayer(hidden1,      DISC_KERNEL,  HIDDEN_SIZE_2, 'gen_conv2')
            hidden3 =    convLayer(hidden2,      DISC_KERNEL,  HIDDEN_SIZE_3, 'gen_conv3')
            hidden4 =    convLayer(hidden3,      DISC_KERNEL,  HIDDEN_SIZE_4, 'gen_conv4')
            hidden_out = convLayer(hidden4,      DISC_KERNEL,  HIDDEN_SIZE_5, 'gen_conv_out')

            # DECONVOLVE THAT KERNEL INTO A BIG OL' PIC
            deconv1 = deconvLayer(input_tensor=hidden_out, channels=GEN_SIZE_1,deconv_kernel=GEN_KERNEL,deconv_strides=DECONV_STRIDES,layer_name="deconv1")
            deconv2 = deconvLayer(input_tensor=deconv1,    channels=GEN_SIZE_2,deconv_kernel=GEN_KERNEL,deconv_strides=DECONV_STRIDES,layer_name="deconv2")
            deconv3 = deconvLayer(input_tensor=deconv2,    channels=GEN_SIZE_3,deconv_kernel=GEN_KERNEL,deconv_strides=DECONV_STRIDES,layer_name="deconv3")
            deconv4 = deconvLayer(input_tensor=deconv3,    channels=GEN_SIZE_4,deconv_kernel=GEN_KERNEL,deconv_strides=DECONV_STRIDES,layer_name="deconv4")
            deconv_out = deconvLayer(input_tensor=deconv4,    channels=NUM_CHANNELS,deconv_kernel=GEN_KERNEL,deconv_strides=DECONV_STRIDES,layer_name="deconv_out")

            # Adding an additional pair of layers
            # That will upscale the image beyond
            # the size requirements then scale
            # it back down, hopefully resulting
            # in greater detail/sharpness
            scale_up = deconvLayer(input_tensor=deconv_out, channels=NUM_CHANNELS,deconv_kernel=GEN_KERNEL,deconv_strides=DECONV_STRIDES,layer_name="deconv_downscale")
            # Don't apply normalization (batch)
            # to output layer
            final = tf.layers.conv2d(scale_up, NUM_CHANNELS, DISC_KERNEL, strides=(2,2), padding='same', activation=tf.nn.tanh)
            #final = tf.layers.conv2d_transpose(inputs=deconv4,filters=NUM_CHANNELS,kernel_size=GEN_KERNEL,strides=DECONV_STRIDES,padding='same',activation=tf.nn.tanh)
            image_shaped_gen= tf.reshape(final,[-1,IMG_SIZE1, IMG_SIZE2, NUM_CHANNELS])
            tf.summary.image('generated_input', image_shaped_gen, NUM_SUMMARY)
            return image_shaped_gen


        with tf.variable_scope("generator") as scope:
            fake_data = generatorDeconv(x)
            #fake_data = generator(g)

        with tf.variable_scope("discriminator") as scope:
            squared_diff = tf.squared_difference(image_shaped_input, fake_data)

        # Define loss function(s)
        with tf.name_scope('loss'):
            generator_loss = tf.reduce_mean(squared_diff)
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
