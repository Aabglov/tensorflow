# IMPORTS
import random
import os
import numpy as np # Used for One-hot encoding


class Vocabulary:
    def __init__(self,vocab=None,custom_go=u'\xbb' ,custom_unk=u'\xac' ,custom_eos=u'\xa4'):
        if not vocab:
            # DEFAULT MAGIC THE GATHERING VOCABULARY
            self.vocab = [u'\xbb','|', '5', 'c', 'r', 'e', 'a', 't', 'u', '4', '6', 'h', 'm', 'n', ' ', 'o', 'd', 'l', 'i', '7', \
                     '8', '&', '^', '/', '9', '{', 'W', '}', ',', 'T', ':', 's', 'y', 'b', 'f', 'v', 'p', '.', '3', \
                     '0', 'A', '1', 'w', 'g', '\\', 'E', '@', '+', 'R', 'C', 'x', 'B', 'G', 'O', 'k', '"', 'N', 'U', \
                     "'", 'q', 'z', '-', 'Y', 'X', '*', '%', '[', '=', ']', '~', 'j', 'Q', 'L', 'S', 'P', '2',u'\xac',u'\xa4']
        else:
            self.vocab = vocab

        # Set characters
        self.go_char = custom_go
        self.unk_char = custom_unk
        self.eos_char = custom_eos
        for c in [custom_go,custom_unk,custom_eos]:
            if c not in self.vocab:
                self.vocab.append(c)
        self.eos = self.vocab.index(self.eos_char)
        self.go = self.vocab.index(self.go_char)
        self.unk = self.vocab.index(self.unk_char)

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
        self.current_batch = self.batches[0]
        # The letter index only applies to
        # this particular dataset
        self.letter_index = 0
        self.num_batches = len(self.batches)


    def next_card_id(self, num_steps):
        x = np.array([[self.vocab.char2id(b) for b in self.current_batch[i:i+num_steps]] for i in range(len(self.current_batch) - num_steps)])
        # The double brackets here ensure we get an array of shape (?,) instead of (1,)
        y = np.array([self.vocab.char2id(self.current_batch[i]) for i in range(num_steps,len(self.current_batch))])

        # Increase the batch index
        self.batch_index += 1
        self.batch_index = self.batch_index % self.num_batches
        self.current_batch = self.batches[self.batch_index]
        return x,y

    def next_card(self):
        # X is a one-hot encoded vector of the corresponding card
        # Y is also an one-hot encoded vector (of the output card)
        x = [self.vocab.id2onehot(self.vocab.char2id(b)) for b in self.current_batch[:-1]] # Don't inclue the EOS tag
        y = [self.vocab.id2onehot(self.vocab.char2id(b)) for b in self.current_batch[1:]]  # Don't include the GO tag

        # Increase the batch index
        self.batch_index += 1
        self.batch_index = self.batch_index % self.num_batches
        self.current_batch = self.batches[self.batch_index]
        return x,y

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
    def __init__(self,raw_text,vocab_list=None, custom_go=u'\xbb' ,custom_unk=u'\xac' ,custom_eos=u'\xa4', split_ratio=[0.7,0.15,0.15]):

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
        else:
            self.vocab_list = vocab_list

        # for each row in raw_txt
        for r in rows:
            print("row {} of {} complete".format(count,total))
            count += 1
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
        self.vocab = Vocabulary(self.vocab_list,custom_go,custom_unk,custom_eos) # Pass our custom tags

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


if __name__ == "__main__":
    mtg_test()
