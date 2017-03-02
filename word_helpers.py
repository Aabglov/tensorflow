# IMPORTS
import random
import os

class WordHelper:
    def __init__(self,raw_text,vocab=None,max_word_size=10, custom_go=u'\xbb' ,custom_unk=u'\xac' ,custom_eos=u'\xa4'):

        self.batches = []
        # Generate vocabulary
        count = 1
        rows = raw_text.split('\n')
        total = len(rows)
        if not vocab:
            # Create our character vocaulary
            self.vocab = []
        else:
            self.vocab = vocab

        # for each row in raw_txt
        for r in rows:
            print("row {} of {} complete".format(count,total))
            count += 1
            # for each letter in that row
            for l in r:
                if l not in self.vocab:
                    self.vocab.append(l)

            # Skip empty lines
            if r.replace(" ","") != "":
                # Add our go and eos tags
                if r[-1] != custom_eos:
                    r = custom_go + r + custom_eos
                else:
                    r = custom_go + r
                self.batches.append(r)
            else:
                # Don't empty lines to our batches
                pass
        else:
            self.vocab = vocab

        # Determine/Set the GO, UNKOWN and END-OF-SEQUENCE tags
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
        # This value may not be used in a character RNN
        self.max_word_size = max_word_size

        # Set up batch generator
        self.batch_index = 0
        self.current_batch = self.batches[0]
        self.letter_index = 0
        self.num_batches = len(self.batches)

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

    def GenBatch(self):
        x = self.char2id(self.current_batch[self.letter_index])
        y = self.char2id(self.current_batch[self.letter_index+1])
        # If we've reached the EOS tag then move onto the next batch
        # NOTE: we're comparing y, not x.
        # There's nothing to generate after y tag == EOS
        if y == self.eos:
            self.letter_index = 0
            self.batch_index += 1
            self.batch_index = self.batch_index % self.num_batches
        # Otherwise just move onto the next letter within this batch
        else:
            self.letter_index += 1
        return x,y





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
    for _ in range(250000):
        a,b = wh.GenBatch()
        print(wh.id2char(a),wh.id2char(b))


if __name__ == "__main__":
    mtg_test()
