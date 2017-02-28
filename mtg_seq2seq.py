import tensorflow as tf
import numpy as np
import data_utils
import os
import word_helpers


# parameters
xseq_len = 8
yseq_len = 8
batch_size = 32
xvocab_size = 26
yvocab_size = 26
emb_dim = 10

# Batch Generator -- Randomly generated so we don't need individual data sets
def genRandWord(self):
    word_len = np.random.randint(1,xvocab_size+1) # randint selects from 1 below high, increase by 1 to accomodate
    word = [self.id2char(np.random.randint(1,self.vocab_size)) for _ in range(word_len)]
    return ''.join(word)

import seq2seq_wrapper

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/twitter/',
                               emb_dim=emb_dim,
                               #epochs=10,
                               num_layers=3
                               )


# In[8]:

val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)


# In[9]:
#sess = model.restore_last_session()
sess = model.train(train_batch_gen, val_batch_gen)

x,y = train_batch_gen.__next__()
# Testing output:
print("INPUT: {}".format(x))
out = model.predict(sess,x)
print("OUTPUT: {}".format(out))
