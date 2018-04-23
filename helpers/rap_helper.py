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



class SongBatcher:
    def __init__(self,songs,vocab):
        # Add data
        self.songs = [s.split("\n") for s in songs] # Turn each song into a list of lines
        self.num_batches = sum([len(s) for s in songs])
        # Set up vocabulary
        self.vocab = Vocabulary(vocab)
        # Set up batch generator
        self.song_index = 0
        self.num_songs = len(self.songs)
        # index within song
        self.batch_index = 0

        for s in self.songs: # Add GO and EOS char to each song
            s = [self.vocab.go_char] + s + [self.vocab.eos_char]

    def current_song(self):
        return self.songs[self.song_index]

    def current_batch(self):
        return self.current_song()[self.batch_index]

    def next(self,max_len=100):
        next_batch_raw = self.songs[self.song_index][self.batch_index]
        next_batch = [self.vocab.char2id(n) for n in next_batch_raw]
        if len(next_batch) < max_len:
            pad_len = (max_len - len(next_batch))
            next_batch += [self.vocab.pad] * pad_len
        new_song = False
        self.batch_index += 1
        if self.batch_index >= len(self.current_song()):
            self.batch_index = 0
            self.song_index = (self.song_index + 1) % self.num_songs
            new_song = True

        return next_batch, new_song
