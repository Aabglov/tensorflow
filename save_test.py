# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import tensorflow as tf

# Where we're saving our data
dir_path = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(dir_path,"saved")

# Attempt to open the saved model/variables
try:
    # Create a session to do our work in
    with tf.Session() as sess:
        # Import meta graph
        new_saver = tf.train.import_meta_graph(os.path.join(save_path,'test.meta'))
        # Restore from path
        new_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(save_path,"test")))
        # Access all values in our collection
        all_vars = tf.get_collection('vars')
        # Do something with them.
        #   In this case, just print them.
        for v in all_vars:
            v_ = sess.run(v)
            print(v.name,v_)

# If we're unable to open the file (OSError)
# then it's because it doesn't exist and need to
# create it.
except OSError as e:
    # Create a session to do our work in.
    sess = tf.Session()
    # Define some variables -- to be printed upon reload
    w1 = tf.Variable(tf.truncated_normal(shape=[10]), name='w1')
    w2 = tf.Variable(tf.truncated_normal(shape=[20]), name='w2')
    # Initialize a Saver object to save our model
    saver = tf.train.Saver()
    # Remember the training_op we want to run by adding it to a collection.
    tf.add_to_collection('vars', w1)
    tf.add_to_collection('vars', w2)
    # Initialize all variables in graph
    sess.run(tf.global_variables_initializer())
    # Save the model to our path
    saver.save(sess, os.path.join(save_path,"test"))
