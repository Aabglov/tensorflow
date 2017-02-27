# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
import collections


url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('data/text8.zip', 31344016)


def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return tf.compat.as_str(f.read(name))
  f.close()

text = read_data(filename)
print('Data size %d' % len(text))

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0

def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '

print(char2id('a'), char2id('z'), char2id(' '), char2id(':'))
print(id2char(1), id2char(26), id2char(0))

batch_size=64
gram_size = 20
num_unrollings=10

class BatchGenerator(object):
  def __init__(self, text, batch_size, gram_size, num_unrollings):
    self._text = text
    self._text_size = len(text) - gram_size
    self._batch_size = batch_size
    self._gram_size = gram_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // (batch_size)
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()

  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size,self._gram_size), dtype=np.float)
    for b in range(self._batch_size):
      for g in range(self._gram_size):
        batch[b,g] = char2id(self._text[self._cursor[b]+g])
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch

  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches

def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (mostl likely) character representation."""
  return [[id2char(int(c)) for c in p] for p in probabilities]

def characters_onehot(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (mostl likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]

def ids_onehot(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (mostl likely) character id"""
  return np.array([embed2onehot(np.array([c])).reshape((vocabulary_size)) for c in np.argmax(probabilities, 1)])

def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    #s = [''.join(x) for x in zip(s,characters(b))]
    s = [''.join(x) for x in zip(s,[c[0] for c in characters(b)])]
  return s

train_batches = BatchGenerator(train_text, batch_size, gram_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, gram_size, 1)

print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))
print(batches2string(valid_batches.next()))
print(batches2string(valid_batches.next()))

def logprob(pred, labels):
  """Log-probability of the true labels in a predicted batch."""
  pred[pred < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(pred))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def sample_gram(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[gram_size, vocabulary_size], dtype=np.float)
  for g in range(gram_size):
    p[g, sample_distribution(prediction[g])] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[gram_size, vocabulary_size])
  return b/np.sum(b, 1)[:,None]

def embed2onehot(embed):
  b = embed.shape[0]
  oh = np.zeros(shape=(b, vocabulary_size), dtype=np.float32)
  for o in range(b):
    oh[o, int(embed[o])] = 1.0
  return oh

num_nodes = 64
embedding_size = 64
# Bigram
gram = 2

graph = tf.Graph()
with graph.as_default():

  # Parameters:
  # Input gate: input, previous output, and bias.
  x_all = tf.Variable(tf.truncated_normal([gram_size*vocabulary_size, 4*num_nodes], -0.1, 0.1))
  m_all = tf.Variable(tf.truncated_normal([num_nodes, 4*num_nodes], -0.1, 0.1))
  ib = tf.Variable(tf.zeros([1, num_nodes]))
  # Forget gate: input, previous output, and bias.
  fb = tf.Variable(tf.zeros([1, num_nodes]))
  # Memory cell: input, state and bias.
  cb = tf.Variable(tf.zeros([1, num_nodes]))
  # Output gate: input, previous output, and bias.
  ob = tf.Variable(tf.zeros([1, num_nodes]))
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
  # Embeddings
  embeddings = tf.Variable(tf.random_uniform([vocabulary_size, vocabulary_size], -1.0, 1.0))
  keep_prob = tf.placeholder(tf.float32)

  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates."""
    i_mul = tf.matmul(i,x_all)
    o_mul = tf.matmul(o,m_all)

    ix_mul = i_mul[:,:num_nodes]# tf.matmul(i, ix)
    fx_mul = i_mul[:,num_nodes:2*num_nodes]# tf.matmul(i, fx)
    cx_mul = i_mul[:,2*num_nodes:3*num_nodes]# tf.matmul(i, cx)
    ox_mul = i_mul[:,3*num_nodes:]# tf.matmul(i, ox)

    im_mul = o_mul[:,:num_nodes] # tf.matmul(o,im)
    fm_mul = o_mul[:,num_nodes:2*num_nodes] # tf.matmul(o,fm)
    cm_mul = o_mul[:,2*num_nodes:3*num_nodes] # tf.matmul(o,cm)
    om_mul = o_mul[:,3*num_nodes:] # tf.matmul(o,om)

    input_gate = tf.sigmoid(ix_mul + im_mul + ib)
    forget_gate = tf.sigmoid(fx_mul + fm_mul + fb)
    update = cx_mul + cm_mul + cb
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(ox_mul + om_mul + ob)
    return output_gate * tf.tanh(state), state

  # Input data.
  train_data = list()
  data_labels = list()
  for _ in range(num_unrollings + 1):
    train_data.append(tf.placeholder(tf.int32, shape=[batch_size,gram_size]))
    data_labels.append(tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
  train_inputs = train_data[:num_unrollings]
  train_labels = data_labels[1:]  # labels are inputs shifted by one time step.

  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    #e = tf.zeros([batch_size,vocabulary_size])
    #for g in range(gram_size):
    #  e += tf.nn.embedding_lookup(embeddings,i[:,g])
    e = tf.nn.embedding_lookup(embeddings,i)
    e_2d = tf.reshape(e,[batch_size,gram_size*vocabulary_size])
    dropout_embed = tf.nn.dropout(tf.identity(e_2d),keep_prob)
    output, state = lstm_cell(dropout_embed, output, state)
    outputs.append(output)

  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.matmul(tf.concat(outputs, 0),w) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.concat(train_labels, 0), logits=logits) )

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    10.0, global_step, 5000, 0.1, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(
    zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits)

  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.int32, shape=[1,gram_size])
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  #embed_sample = tf.zeros([1,vocabulary_size])
  #for g in range(gram_size):
  #  embed_sample += tf.nn.embedding_lookup(embeddings,sample_input[:,g])
  embed_sample = tf.nn.embedding_lookup(embeddings,sample_input)
  embed_sample_2d = tf.reshape(embed_sample,[1,gram_size*vocabulary_size])
  sample_output, sample_state = lstm_cell(embed_sample_2d, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
      feed_dict[data_labels[i]] = embed2onehot(batches[i][:,-1])
      feed_dict[keep_prob] = 0.7
    _, l, predictions, lr = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print(
        'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = embed2onehot(np.concatenate(list([batch[:,-1] for batch in batches])[1:]))
      print('Minibatch perplexity: %.2f' % float(
        np.exp(logprob(predictions, labels))))
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = collections.deque(maxlen=gram_size)
          init_feed = sample_gram(random_distribution())
          for f in init_feed:
            feed.append(f)
          sentence = characters_onehot(feed)[0]
          reset_sample_state.run()
          for _ in range(79):
            feed_search = np.array(feed)
            loc = []
            for f in feed:
              loc.append(np.where(f==1.))
            embed_feed = np.array(loc).reshape((1,gram_size))
            prediction = sample_prediction.eval({sample_input: embed_feed})
            new_feed = sample(prediction)
            feed.append(new_feed.reshape((vocabulary_size,)))
            sentence += characters_onehot(new_feed)[0]
          print(sentence)
        print('=' * 80)
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size):
        batch = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: batch[0]})
        gram_batch = np.array([batch[1][0][-1]])
        valid_logprob = valid_logprob + logprob(predictions, embed2onehot(gram_batch))
      print('Validation set perplexity: %.2f' % float(np.exp(
        valid_logprob / valid_size)))
