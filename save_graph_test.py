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
        new_saver = tf.train.import_meta_graph(os.path.join(save_path,'graph.meta'))
        # Restore from path
        new_saver.restore(sess, os.path.join(save_path,"graph"))
        # Access all values in our collection
        all_vars = tf.get_collection('var')
        # Do something with them.
        #   In this case, just print them.
        for v in all_vars:
            v_ = sess.run(v)
            print(v.name,v_)

# If we're unable to open the file (OSError)
# then it's because it doesn't exist and need to
# create it.
except OSError as e:
    # Create a graph to perform calculations
    graph = tf.Graph()
    with graph.as_default():
        variable = tf.Variable(42, name='foo')
        # Initialize all variables in graph
        initialize = tf.global_variables_initializer()
        assign = variable.assign(13)
        graph.add_to_collection('var',variable)

    # Create a session to do our work in.
    with tf.Session(graph=graph) as sess:
        # Initialize a Saver object to save our model
        saver = tf.train.Saver()
        sess.run(initialize)
        sess.run(assign)
        print(sess.run(variable))
        v = graph.get_collection('var')[0]
        # Save the model to our path
        saver.save(sess, os.path.join(save_path,"graph"))
