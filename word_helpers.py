# SCIPY
import random

class WordHelper:
    def __init__(self,vocab,max_word_size=10, custom_go=u'\xbb' ,custom_unk=u'\xac' ,custom_eos=u'\xa4'):
        # Determine/Set the GO, UNKOWN and END-OF-SEQUENCE tags
        self.go_char = custom_go
        self.unk_char = custom_unk
        self.eos_char = custom_eos
        for c in [custom_go,custom_unk,custom_eos]:
            if c not in vocab:
                vocab.append(c)
        self.eos = vocab.index(self.eos_char)
        self.go = vocab.index(self.go_char)
        self.unk = vocab.index(self.unk_char)

        # Set values to class
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.vocab_indices = [i for i in range(self.vocab_size)]
        # This value may not be used in a character RNN
        self.max_word_size = max_word_size

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

    # Word helpers
    def genRandWord(self):
        word_len = np.random.randint(1,self.max_word_size+1) # randint selects from 1 below high, increase by 1 to accomodate
        word = [self.id2char(np.random.randint(1,self.vocab_size)) for _ in range(word_len)]
        return ''.join(word)

    def genRandBatch(self):
        word = self.genRandWord()
        batch = self.word2batch(word)
        rev_batch = self.word2batch(self.reverseWord(word))
        return castInt(batch),castInt(rev_batch)

    # BATCH CONVERSIONS
    def batch2word(self,batch):
        return ''.join([self.id2char(i) for i in batch if i != self.eos]) # Skip End of Sequence tag

    def word2batch(self,word):
        batch = [self.char2id(letter) for letter in word] + [self.eos] # Add End of Sequence tag
        return batch

    def id2onehot(self,i):
        oh = np.zeros(self.vocab_size)
        oh[i] = 1
        return oh

    def onehot2id(self,oh):
        i = np.argmax(oh)
        return i

    def words2text(self,words):
        text = ''
        for w in words:
            if len(w) > self.max_word_size:
                words.remove(w)
            else:
                text += w + (' ' * (self.max_word_size - len(w)))
        return text
