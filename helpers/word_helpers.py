# IMPORTS
import random
import os
import numpy as np # Used for One-hot encoding
import json # Used for json helper

class Vocabulary:
    def __init__(self,vocab=None,custom_go=u'\xbb' ,custom_unk=u'\xac' ,custom_pad=u'\xf8', custom_eos=u'\xa4', custom_split=u'\u00BB'):
        if not vocab:
            # DEFAULT MAGIC THE GATHERING VOCABULARY
            self.vocab = [u'\xbb','|', '5', 'c', 'r', 'e', 'a', 't', 'u', '4', '6', 'h', 'm', 'n', ' ', 'o', 'd', 'l', 'i', '7', \
                     '8', '&', '^', '/', '9', '{', 'W', '}', ',', 'T', ':', 's', 'y', 'b', 'f', 'v', 'p', '.', '3', \
                     '0', 'A', '1', 'w', 'g', '\\', 'E', '@', '+', 'R', 'C', 'x', 'B', 'G', 'O', 'k', '"', 'N', 'U', \
                     "'", 'q', 'z', '-', 'Y', 'X', '*', '%', '[', '=', ']', '~', 'j', 'Q', 'L', 'S', 'P', '2',u'\xac', u'\xf8', u'\xa4',u'\u00BB']
        else:
            self.vocab = vocab

        # Set characters
        self.go_char = custom_go
        self.unk_char = custom_unk
        self.pad_char = custom_pad
        self.eos_char = custom_eos
        self.split_char = custom_split
        for c in [custom_go,custom_unk,custom_pad,custom_eos,custom_split]:
            if c not in self.vocab:
                self.vocab.append(c)
        self.eos = self.vocab.index(self.eos_char)
        self.go = self.vocab.index(self.go_char)
        self.pad = self.vocab.index(self.pad_char)
        self.unk = self.vocab.index(self.unk_char)
        self.split = self.vocab.index(self.split_char)

        # Set values to class
        self.vocab_size = len(self.vocab)
        self.vocab_indices = [i for i in range(self.vocab_size)]

    def char2id(self,char):
        if char in self.vocab:
            return self.vocab.index(char)
        else:
            return self.unk

    def id2char(self,dictid):
        if dictid in self.vocab_indices:
            return self.vocab[dictid]
        else:
            return self.unk_char

    def id2onehot(self,i):
        oh = np.zeros((1,self.vocab_size))
        oh[0,i] = 1
        return oh

    def onehot2id(self,oh):
        i = np.argmax(oh)
        return i

    def char2onehot(self,char):
        return self.id2onehot(self.char2id(char))

    def onehot2char(self,oh):
        return self.id2char(self.onehot2id(oh))



# NOTE: This can be a batch generator for any data set,
# Training, Test or Cross-Validation
class BatchGenerator:
    def __init__(self,data_rows,vocab):
        # Add data
        self.batches = data_rows
        # Set up vocabulary
        assert type(vocab).__name__ == "Vocabulary", "Vocabulary object is not class, but is instead: {}".format(type(vocab).__name__)
        self.vocab = vocab # Vocabulary class
        # Set up batch generator
        self.batch_index = 0
        self.current_batch = self.batches[0] # single batch generation
        self.current_batches = [] # multiple batch prediction
        # The letter index only applies to
        # this particular dataset
        self.letter_index = 0
        self.num_batches = len(self.batches)


    def next_card_batch(self, batch_size, num_steps=1):
        # This will be the maximum batch length among all our batches
        max_len = 0
        # Add batch_size number of batches to our current_batches list
        for b in range(batch_size):
            # Set maximum batch lenght
            max_len = max(max_len,len(self.current_batch))
            self.current_batches.append(self.current_batch)
            self.batch_index += 1
            self.batch_index = self.batch_index % self.num_batches
            self.current_batch = self.batches[self.batch_index]

        # Ensure our max_len is divisible by num_steps
        if num_steps > 1:
            max_len += num_steps - (max_len % num_steps)

        # Add Spacer
        # This one is a little weird at first, but it makes sense.
        # We end up iterating over this batch by NUM_STEPS and
        # because we need the "next" letter to populate
        # our target (y) value we need an extra (+1).
        # The second +1 comes from the way I've set up the loop
        # and the fact that python is 0-indexed.
        # See mtg_rec_char.py for a better idea.
        max_len += 2

        batch_collection = []
        for b in self.current_batches:
            batch = []
            for i in range(max_len):
                try:
                    # Attempt to get the character in batch b at position i
                    c = b[i]
                # If this fails it's because max_len is longer than this batch.
                # In this case we'll return a padding character to go after
                # the EOS
                except IndexError as e:
                    c = self.vocab.pad_char
                # Add the character to the batch
                batch.append(self.vocab.char2id(c))
            batch_collection.append(batch)

        # Reset the current_batches.
        self.current_batches = []
        return np.array(batch_collection)

    def next(self):
        # X is a one-hot encoded vector of the corresponding character
        # Y is also an one-hot encoded vector (of the output character)
        x = self.vocab.id2onehot(self.vocab.char2id(self.current_batch[self.letter_index]))
        y = self.vocab.id2onehot(self.vocab.char2id(self.current_batch[self.letter_index+1]))
        # If we've reached the end of the word
        # (this would be the EOS tag)
        # then move onto the next batch because
        # there's nothing to generate after y tag == EOS.
        #
        # NOTE: We could just check to see if y == eos here
        # but it's possible that we fuck up and there isn't an
        # EOS tag at the end of the batch.
        # That SHOULDN'T happen and would be an error,
        # but still.  Safer to check the index.
        if self.letter_index == len(self.current_batch) - 2:
            #                                            ^^^
            # The minus 2 refers to the penultimate element of the
            # batch since we can't perform prediction on the very last element
            self.letter_index = 0
            self.batch_index += 1
            self.batch_index = self.batch_index % self.num_batches
            self.current_batch = self.batches[self.batch_index]
        # Otherwise just move onto the next letter within this batch
        else:
            self.letter_index += 1
        return x,y


class WordHelper:
    def __init__(self,raw_text,vocab_list=None, custom_go=u'\xbb' ,custom_unk=u'\xac' ,custom_pad=u'\xf8' ,custom_eos=u'\xa4', split_ratio=[0.7,0.15,0.15]):

        self.train_batches = []
        self.test_batches = []
        self.valid_batches = []
        # Generate vocabulary
        count = 1
        rows = raw_text.split('\n')
        total = len(rows)

        # Determine the indices by which we'll
        # split our data
        train_end = int(total * split_ratio[0])
        test_end = train_end + int(total * split_ratio[1])
        if not vocab_list:
            # Create our character vocaulary
            self.vocab_list = []
            self.provided_vocab = False
        else:
            self.vocab_list = vocab_list
            self.provided_vocab = True

        # for each row in raw_txt
        for r in rows:
            print("row {} of {} complete".format(count,total))
            count += 1
            # for each letter in that row
            if not self.provided_vocab:
                for l in r:
                    if l not in self.vocab_list:
                        self.vocab_list.append(l)

            # Skip empty lines
            if r.replace(" ","") != "":
                # Add our go and eos tags
                if r[-1] != custom_eos:
                    r = custom_go + r + custom_eos
                else:
                    r = custom_go + r

                if count < train_end:
                    self.train_batches.append(r)
                elif count >= train_end and count < test_end:
                    self.test_batches.append(r)
                elif count >= test_end:
                    self.valid_batches.append(r)

            else:
                # Don't empty lines to our batches
                pass

        # Create our vocabulary object
        self.vocab = Vocabulary(self.vocab_list,custom_go,custom_unk,custom_pad,custom_eos) # Pass our custom tags

        # All these batches tryin' to front!
        self.TrainBatches = BatchGenerator(self.train_batches,self.vocab)
        self.TestBatches = BatchGenerator(self.test_batches,self.vocab)
        self.ValidBatches = BatchGenerator(self.valid_batches,self.vocab)


class JSONHelper:
    def __init__(self,filepath,vocab_list=None, custom_go=u'\xbb' ,custom_unk=u'\xac' ,custom_pad=u'\xf8' ,custom_eos=u'\xa4', custom_split=u'\u00BB', split_ratio=[0.8,0.1,0.1]):

        self.train_batches = []
        self.test_batches = []
        self.valid_batches = []
        # Generate vocabulary
        count = 1
        with open(filepath,"r") as f:
            data = json.load(f)

        total = len(data)

        # Determine the indices by which we'll
        # split our data
        train_end = int(total * split_ratio[0])
        test_end = train_end + int(total * split_ratio[1])
        if not vocab_list:
            # Create our character vocaulary
            self.vocab_list = []
        else:
            self.vocab_list = vocab_list

        # for each row in raw_txt
        for k,v in data.items():
            print("row {} of {} complete".format(count,total))
            count += 1
            r = "{}{}{}".format(k,custom_split,v)
            # for each letter in that row
            for l in r:
                if l not in self.vocab_list:
                    self.vocab_list.append(l)

            # Skip empty lines
            if r.replace(" ","") != "":
                # Add our go and eos tags
                if r[-1] != custom_eos:
                    r = custom_go + r + custom_eos
                else:
                    r = custom_go + r

                if count < train_end:
                    self.train_batches.append(r)
                elif count >= train_end and count < test_end:
                    self.test_batches.append(r)
                elif count >= test_end:
                    self.valid_batches.append(r)

            else:
                # Don't empty lines to our batches
                pass

        # Create our vocabulary object
        self.vocab = Vocabulary(self.vocab_list,custom_go,custom_unk,custom_pad,custom_eos,custom_split) # Pass our custom tags

        # All these batches tryin' to front!
        self.TrainBatches = BatchGenerator(self.train_batches,self.vocab)
        self.TestBatches = BatchGenerator(self.test_batches,self.vocab)
        self.ValidBatches = BatchGenerator(self.valid_batches,self.vocab)





def mtg_test():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path,"saved","mtg","mtg.ckpt")
    data_path = os.path.join(dir_path,"data","cards_tokenized.txt")

    # Load mtg tokenized data
    # Special thanks to mtgencode: https://github.com/billzorn/mtgencode
    with open(data_path,"r") as f:
        # Each card occupies its own line in this tokenized version
        raw_txt = f.read()

    wh = WordHelper(raw_txt)
    for _ in range(25000):
        a,b = wh.TrainBatches.next()
        print(wh.vocab.onehot2char(a),wh.vocab.onehot2char(b))

    for _ in range(2500):
        a,b = wh.TestBatches.next()
        print(wh.vocab.onehot2char(a),wh.vocab.onehot2char(b))

    for _ in range(2500):
        a,b = wh.ValidBatches.next()
        print(wh.vocab.onehot2char(a),wh.vocab.onehot2char(b))

def json_test():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path,"data","dictionary.json")

    jh = JSONHelper(data_path)
    for _ in range(2500):
        a,b = jh.TrainBatches.next()
        print(jh.vocab.onehot2char(a),jh.vocab.onehot2char(b))

    for _ in range(250):
        a,b = jh.TestBatches.next()
        print(jh.vocab.onehot2char(a),jh.vocab.onehot2char(b))

    for _ in range(250):
        a,b = jh.ValidBatches.next()
        print(jh.vocab.onehot2char(a),jh.vocab.onehot2char(b))


if __name__ == "__main__":
    #mtg_test()
    json_test()
