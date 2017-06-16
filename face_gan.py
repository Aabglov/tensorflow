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

if tf.gfile.Exists(LOG_DIR):
    tf.gfile.DeleteRecursively(LOG_DIR)
tf.gfile.MakeDirs(LOG_DIR)

######################################### CONSTANTS ########################################
NUM_CLASSES = 10 # The number of digits present in the dataset
DEVICE = "/gpu:0" # Controls whether we run on CPU or GPU
NUM_CHANNELS = 3
DIS_LEARNING_RATE = 0.001
GEN_LEARNING_RATE = 0.001
ORIG_IMG_SIZE1 = 218
ORIG_IMG_SIZE2 = 178

# Resize the images so it doesn't crash my computer
IMG_SIZE1 = 54
IMG_SIZE2 = 44

# GENERATOR
GEN_SIZE_IN = 100
GEN_KERNEL = [20,17]
GEN_SIZE_1 = 50 # 1st layer number of features
GEN_SIZE_2 = 25 # 2nd layer number of features
GEN_SIZE_3 = 3 # final layer

# DISCRIMINATOR
HIDDEN_SIZE_1 = 32 # 1st layer number of features
HIDDEN_SIZE_2 = 64 # 2nd layer number of features
HIDDEN_SIZE_3 = 128 # 3rd layer
HIDDEN_SIZE_4 = 256 # Dense layer -- ouput
KERNEL_SIZE_1 = [10,10]
KERNEL_SIZE_2 = [5,5]

BATCH_SIZE = 100
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
# input_tensor = init_weight([1,SMALL_IMG_SIZE1,SMALL_IMG_SIZE2,3],"input")
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
#
# gen_input = init_weight([1,1,1,100],"genput")
# deconv1 = tf.layers.conv2d_transpose(inputs=gen_input,filters=50,kernel_size=[10,10],strides=(1,1),activation=tf.nn.relu)
# deconv2 = tf.layers.conv2d_transpose(inputs=deconv1,  filters=25,kernel_size=[5,5],strides=(2,2),activation=tf.nn.relu)
# deconv3 = tf.layers.conv2d_transpose(inputs=deconv2,  filters=3,kernel_size=[5,5],strides=(2,2),activation=tf.nn.sigmoid)
# flat = tf.contrib.layers.flatten(deconv3)
# dense = tf.layers.dense(inputs=flat, units=SMALL_IMG_SIZE1*SMALL_IMG_SIZE2*NUM_CHANNELS, activation=tf.identity)
# final = tf.reshape(dense,[-1,SMALL_IMG_SIZE1, SMALL_IMG_SIZE2, NUM_CHANNELS])
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
        image_std = tf.image.per_image_standardization(image_orig)
        image = tf.image.resize_images(image_std, [ORIG_IMG_SIZE1, ORIG_IMG_SIZE2])
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
            g = tf.placeholder(tf.float32, [None, GEN_SIZE_IN] , name="generator_input") # Random input vector
            g_shaped = tf.reshape(g,[-1,1,1,GEN_SIZE_IN])
            #y = tf.placeholder(tf.int32, [None,], name='y-input') # Labels

        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(x,[-1,IMG_SIZE1, IMG_SIZE2, NUM_CHANNELS])
            tf.summary.image('input', image_shaped_input, NUM_CLASSES)
            #shaped_labels = tf.reshape(tf.one_hot(y,NUM_CLASSES),[-1,NUM_CLASSES])

        def convLayer(input_tensor, kernel_shape, channel_dim, output_dim, layer_name, dr=0.2, pool_size=3, act=tf.nn.sigmoid):
            with tf.variable_scope(layer_name) as scope:
                # 2D Convolution
                conv = tf.layers.conv2d(input_tensor,channel_dim,kernel_shape,strides=(1,1),padding='same',activation=act)
                # Pooling
                pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[pool_size,pool_size], strides=pool_size)
                dropout = tf.layers.dropout(inputs=pool, rate=dr)
                return dropout

        # DEFINE GENERATOR USING DECONVOLUTION
        def generatorDeconv(gen_input):
            deconv1 = tf.layers.conv2d_transpose(inputs=gen_input,filters=GEN_SIZE_1,kernel_size=KERNEL_SIZE_1,strides=(1,1),activation=tf.nn.relu)
            deconv2 = tf.layers.conv2d_transpose(inputs=deconv1,  filters=GEN_SIZE_2,kernel_size=KERNEL_SIZE_2,strides=(2,2),activation=tf.nn.relu)
            deconv3 = tf.layers.conv2d_transpose(inputs=deconv1,  filters=GEN_SIZE_3,kernel_size=KERNEL_SIZE_2,strides=(2,2),activation=tf.nn.sigmoid)
            flat = tf.contrib.layers.flatten(deconv3)
            dense = tf.layers.dense(inputs=flat, units=IMG_SIZE1*IMG_SIZE2*NUM_CHANNELS, activation=tf.identity)
            image_shaped_gen= tf.reshape(dense,[-1,IMG_SIZE1, IMG_SIZE2, NUM_CHANNELS])
            tf.summary.image('generated_input', image_shaped_gen, NUM_CLASSES)
            #return gen2
            return image_shaped_gen

        # DEFINE DISCRIMINATOR
        def discriminatorConv(input_tensor):
            hidden1 = convLayer(input_tensor, KERNEL_SIZE_1, NUM_CHANNELS,  HIDDEN_SIZE_1, 'layer1' , act=tf.nn.relu)
            hidden2 = convLayer(hidden1,      KERNEL_SIZE_2, HIDDEN_SIZE_1, HIDDEN_SIZE_2, 'layer2' , act=tf.nn.relu)
            hidden_out = convLayer(hidden2,   KERNEL_SIZE_2, HIDDEN_SIZE_2, HIDDEN_SIZE_3, 'layer3')
            # Dense Layer
            with tf.variable_scope("dense") as scope:
                flat = tf.contrib.layers.flatten(hidden_out)
                dense = tf.layers.dense(inputs=flat, units=HIDDEN_SIZE_4, activation=tf.nn.relu)
                # Logits Layer
                dropout = tf.layers.dropout(inputs=dense, rate=0.2)
                logits = tf.layers.dense(inputs=dropout, units=1)
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
            train_d_step = tf.train.AdamOptimizer(DIS_LEARNING_RATE).minimize(discriminator_loss,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator'))
            train_g_step = tf.train.AdamOptimizer(GEN_LEARNING_RATE).minimize(generator_loss,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'))

        # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'test'))

        # Initializing the variables
        init = tf.global_variables_initializer()

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()

    #Running first session
    #def main():
    with tf.Session(graph=graph) as sess:#,config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Initialize variables
        sess.run(init)

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


            G_INPUT = np.random.uniform(-1., 1., size=[BATCH_SIZE,GEN_SIZE_IN])
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
        train_writer.close()
        test_writer.close()
        print("Training Finished!")

        # Save model weights to disk
        save_path = saver.save(sess, SAVE_PATH)
        print("Model saved in file: %s" % save_path)
