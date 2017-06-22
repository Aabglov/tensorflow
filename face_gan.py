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
DIS_LEARNING_RATE = 0.0002 #0.001
GEN_LEARNING_RATE = 0.0002 #0.001
ADAM_BETA = 0.5
ORIG_IMG_SIZE1 = 218
ORIG_IMG_SIZE2 = 178

# Resize the images so it doesn't crash my computer
IMG_SIZE1 = 64
IMG_SIZE2 = 52

# GENERATOR
#GEN_SIZE_IN = 10000
GEN_SIZE_IN1 = 4#20
GEN_SIZE_IN2 = 3#17
GEN_CHANNELS = 1024
GEN_TOTAL_IN = GEN_SIZE_IN1 * GEN_SIZE_IN2 * GEN_CHANNELS
GEN_SIZE_1 = 512 # 1st layer number of features
GEN_SIZE_2 = 256 # 2nd layer number of features
GEN_SIZE_3 = 128 # 3rd layer
GEN_SIZE_4 = 64# final layer
GEN_KERNEL = [2,2]
DECONV_STRIDES = (2,2)
CONV_KERNEL = [2,2]

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
# def init_weight(shape,name,sd=None):
#            """Create a weight variable with appropriate initialization."""
#            if not sd:
#                sd = 1. / tf.sqrt(shape[0] / 2.)
#            initial = tf.truncated_normal(shape, stddev=sd)
#            return tf.get_variable(name="{}_weights".format(name), initializer=initial)
#
# input_tensor = init_weight([1,IMG_SIZE1,IMG_SIZE2,3],"input")
# kernel = init_weight([10,10,3,16],"kernel")
# conv = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding='SAME')
# pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[3,3], strides=3)
#
# kernel2 = init_weight([5,5,16,64],"kernel2")
# conv2 = tf.nn.conv2d(pool, kernel2, [1, 1, 1, 1], padding='SAME')
# pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3,3], strides=3)
#
# kernel3 = init_weight([5,5,64,128],"kernel3")
# conv3 = tf.nn.conv2d(pool2, kernel3, [1, 1, 1, 1], padding='SAME')
# pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 3], strides=3)
#
# gen_input = init_weight([1,4,3,128],"genput")
# deconv1 = tf.layers.conv2d_transpose(inputs=gen_input,filters=256,kernel_size=[2,2],strides=(2,2),activation=tf.nn.relu)
# #conv1 =   tf.layers.conv2d(deconv1,256,[3,3],strides=(2,2),padding='valid',activation=None)
# deconv2 = tf.layers.conv2d_transpose(inputs=deconv1,  filters=256,kernel_size=[2,2],strides=(2,2),activation=tf.nn.relu)
# #conv2 =   tf.layers.conv2d(deconv2,256,[3,3],strides=(2,2),padding='valid',activation=None)
# deconv3 = tf.layers.conv2d_transpose(inputs=deconv2,  filters=128,kernel_size=[2,2],strides=(2,2),activation=tf.nn.relu)
# #conv3 =   tf.layers.conv2d(deconv3,128,[3,3],strides=(2,2),padding='valid',activation=None)
# deconv4 = tf.layers.conv2d_transpose(inputs=deconv3,  filters=3,  kernel_size=[2,2],strides=(2,2),activation=tf.nn.relu)
# #conv4 =   tf.layers.conv2d(deconv4,3,[2,2],strides=(2,2),padding='valid',activation=None)
# flat = tf.contrib.layers.flatten(conv4)
# dense = tf.layers.dense(inputs=flat, units=IMG_SIZE1*IMG_SIZE2*NUM_CHANNELS, activation=tf.identity)
# final = tf.reshape(dense,[-1,IMG_SIZE1, IMG_SIZE2, NUM_CHANNELS])
# HODOR

# Make a queue of file names including all the JPEG images files in the relative
# image directory.
#filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(os.path.join(DATA_PATH,"*.jpg")))

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
#image_reader = tf.WholeFileReader()

# Read a whole file from the queue
#filename, image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
#image = tf.image.decode_jpeg(image_file)

# with tf.Session() as sess:
#     # Required to get the filename matching to run.
#     tf.global_variables_initializer().run()
#
#     # Coordinate the loading of image files.
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     # Get an image tensor and print its value.
#     for _ in range(10):
#         key,image_tensor = sess.run([filename,image])
#         print(k)
#         #print(image_tensor)
#
# HODOR

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
            x = tf.image.resize_images(images,[IMG_SIZE1,IMG_SIZE2])#images
            #g = tf.placeholder(tf.float32, [None, GEN_SIZE_IN] , name="generator_input") # Random input vector
            #g_shaped = tf.reshape(g,[-1,1,1,GEN_SIZE_IN])
            g = tf.placeholder(tf.float32, [None, GEN_TOTAL_IN] , name="generator_input") # Random input vector
            g_shaped = tf.reshape(g,[-1,GEN_SIZE_IN1,GEN_SIZE_IN2,GEN_CHANNELS])

        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(x,[-1,IMG_SIZE1, IMG_SIZE2, NUM_CHANNELS])
            tf.summary.image('input', image_shaped_input, NUM_CLASSES)
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
                conv = tf.layers.conv2d(input_tensor,channel_dim,kernel_shape,strides=(1,1),padding='same',activation=None)
                # Pooling
                #pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[pool_size,pool_size], strides=pool_size)
                # Psuedo down-sampling
                down = tf.layers.conv2d(conv, channel_dim, [pool_size,pool_size], (pool_size,pool_size), padding='valid', activation=None)
                #act_out = act(pool)
                #norm = tf.layers.dropout(inputs=act_out, rate=dr)
                norm = act(tf.layers.batch_normalization(down,momentum=0.9,training=True),name=layer_name, alpha=0.1)
                return norm

        def deconvLayer(input_tensor,channels,deconv_kernel,deconv_strides,conv_kernel,conv_strides,layer_name,act=tf.nn.relu):
            with tf.variable_scope(layer_name) as scope:
                conv =   tf.layers.conv2d(inputs=input_tensor,filters=channels,kernel_size=conv_kernel,strides=conv_strides,padding='same',activation=None)
                deconv = tf.layers.conv2d_transpose(inputs=conv,filters=channels,kernel_size=deconv_kernel,strides=deconv_strides,activation=None)
                norm = act(tf.layers.batch_normalization(deconv,momentum=0.9,training=True))
                return norm

        # DEFINE GENERATOR USING DECONVOLUTION
        def generatorDeconv(gen_in):
            deconv1 = deconvLayer(input_tensor=gen_in ,channels=GEN_SIZE_1,deconv_kernel=GEN_KERNEL,deconv_strides=DECONV_STRIDES,conv_kernel=CONV_KERNEL,conv_strides=(1,1),layer_name="deconv1")
            deconv2 = deconvLayer(input_tensor=deconv1,channels=GEN_SIZE_2,deconv_kernel=GEN_KERNEL,deconv_strides=DECONV_STRIDES,conv_kernel=CONV_KERNEL,conv_strides=(1,1),layer_name="deconv2")
            deconv3 = deconvLayer(input_tensor=deconv2,channels=GEN_SIZE_3,deconv_kernel=GEN_KERNEL,deconv_strides=DECONV_STRIDES,conv_kernel=CONV_KERNEL,conv_strides=(1,1),layer_name="deconv3")
            #deconv4 = deconvLayer(input_tensor=deconv3,channels=GEN_SIZE_4,deconv_kernel=GEN_KERNEL,deconv_strides=(2,2),conv_kernel=CONV_KERNEL,conv_strides=(2,2),layer_name="deconv4")
            deconv_out = deconvLayer(input_tensor=deconv3,channels=NUM_CHANNELS,deconv_kernel=GEN_KERNEL,deconv_strides=DECONV_STRIDES,conv_kernel=CONV_KERNEL,conv_strides=(1,1),layer_name="deconv_out",act=tf.nn.tanh)
            flat = tf.contrib.layers.flatten(deconv_out)
            dense = tf.layers.dense(inputs=flat, units=IMG_SIZE1*IMG_SIZE2*NUM_CHANNELS, activation=tf.nn.relu)
            image_shaped_gen= tf.reshape(dense,[-1,IMG_SIZE1, IMG_SIZE2, NUM_CHANNELS])
            tf.summary.image('generated_input', image_shaped_gen, NUM_CLASSES)
            #return gen2
            return image_shaped_gen

        # DEFINE DISCRIMINATOR
        def discriminatorConv(input_tensor):
            hidden1 =    convLayer(input_tensor, DISC_KERNEL,  HIDDEN_SIZE_1, 'layer1')
            hidden2 =    convLayer(hidden1,      DISC_KERNEL,  HIDDEN_SIZE_2, 'layer2')
            hidden3 =    convLayer(hidden2,      DISC_KERNEL,  HIDDEN_SIZE_3, 'layer3')
            hidden_out = convLayer(hidden3,      DISC_KERNEL,  HIDDEN_SIZE_4, 'layer_out', pool_size=1)
            # Dense Layer
            with tf.variable_scope("dense") as scope:
                flat = tf.contrib.layers.flatten(hidden_out)
                #dense = tf.layers.dense(inputs=flat, units=HIDDEN_SIZE_4, activation=tf.nn.relu)
                # Logits Layer
                #dropout = tf.layers.dropout(inputs=dense, rate=0.2)
                logits = tf.layers.dense(inputs=flat, units=1)
            prob = tf.nn.sigmoid(logits)
            return prob


        with tf.variable_scope("generator") as scope:
            fake_data = generatorDeconv(g_shaped)
            #fake_data = generator(g)

        with tf.variable_scope("discriminator") as scope:
            fake_prob = discriminatorConv(fake_data)
            #fake_prob = discriminator(fake_data)
            scope.reuse_variables()
            real_prob = discriminatorConv(image_shaped_input)
            #real_prob = discriminator(x)

        # Define loss function(s)
        with tf.name_scope('loss'):
            discriminator_loss = -tf.reduce_mean(tf.log(real_prob) + tf.log(1. - fake_prob))
            generator_loss = -tf.reduce_mean(tf.log(fake_prob))
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


            G_INPUT = np.random.uniform(-1., 1., size=[BATCH_SIZE,GEN_TOTAL_IN])
            # Run optimization op (backprop) and cost op (to get loss value)
            summary,img, g_unused, _d,_g = sess.run([merged, image, fake_data, train_d_step, train_g_step], feed_dict={g: G_INPUT})
            train_writer.add_summary(summary, epoch)
            print('Adding run data for', epoch)


            # Display logs per epoch step
            if epoch % LOG_FREQUENCY == 0:
                #   I'm not a billion percent sure what this does....
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, img, _d,_g = sess.run([merged, image, train_d_step, train_g_step],
                                      feed_dict={g: G_INPUT},
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
