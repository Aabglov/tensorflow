# IMPORTS
import random
import os
import numpy as np # Used for One-hot encoding
import json # Used for json helper
import ast # USed to parse List from string ("['a','b',c']")
#import caffeine
import io

import re

EOS_SEARCH = "[\.,?!]+"
REGEX_SEARCH = '[^0-9a-zA-Z.,?!]+'
GO = u'\xbb'
UNK = u'\xbf'
PAD = u'\xac'
EOS = u'\xf8'
SPLIT = u'\xa4'

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(DIR_PATH,"data","dialog")
STRUCTURE_PATH = os.path.join(DATA_PATH,"movie_conversations.txt")
LINES_PATH = os.path.join(DATA_PATH,"movie_lines.txt")
SAVE_PATH = os.path.join(DATA_PATH,"parsed")
SPACER = " +++$+++ " # Arbitrary spacer token used by dataset
DEBUG = False
#DEBUG = True
REMOVE_SINGLES = False

SYMBOLS = ["~", "`", "!", "<", ">", ".", ",", \
           ":", ";", "\"", "'", "\\", "/", "(", \
           ")", "[", "]", "^", "?", "-", "+", \
           "{", "}", "&", "'",\
           # Separate numbers into their own entries for vocabulary.
           # We don't want to have every 3 digit number as its own word.
           "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

VOCAB_TOP = 30000 

class Conversation:
    def __init__(self,subject1,subject2,lines):
        self.subject1 = subject1
        self.subject2 = subject2
        self.lines = lines

def preTokenize(text):
    # Separate symbols into their own entries
    # in our vocabulary.
    # eg "you." will become "you" and "."

    for s in SYMBOLS:
        text = text.replace(s," {} ".format(s))

    return text.strip()

def indexEncode(word,vocabulary):
    if word in vocabulary:
        return vocabulary.index(word)
    else:
        return UNK

def parseDialog():
    convs = []
    lines = {}
    vocabulary = [GO ,UNK ,PAD ,EOS ,SPLIT] + SYMBOLS
    vocab_occur = {}

    input_seq = []
    target_seq = []

    print("reading lines...")
    with io.open(LINES_PATH,"r",encoding="latin1") as f:
        lines_raw = f.read().split("\n")

    for line_raw in lines_raw:
        if line_raw != "":
            line_parts = line_raw.split(SPACER)
            text = line_parts[4].lower()
            lines[line_parts[0]] = text

            # While we're here, we minds as well
            # create a vocabulary for word embedding
            # some characters are separators in addition
            # to the space character " ".  We
            # handle that here:
            text = preTokenize(text)

            cleaned_text = re.sub(REGEX_SEARCH, ' ', text)
            tokens = cleaned_text.strip().split(" ")
            for token in tokens:
                t = preTokenize(token)
                # This will add duplicates, but
                # we remove them in the next step
                vocabulary.append(t)
                # Count occurences of each word
                if t not in vocab_occur:
                    vocab_occur[t] = 1
                else:
                    vocab_occur[t] += 1

    print("removing duplicates...")
    # Remove duplicates
    vocabulary = list(set(vocabulary))

    if REMOVE_SINGLES:
        for k,v in vocab_occur.items():
            if v <= 1:
                vocabulary.remove(k)

    # Remove null from vocabulary
    vocabulary.remove("")
    print(len(vocabulary))

    if VOCAB_TOP > 0:
        sorted_vocab_occur = sorted(vocab_occur.items(), key=lambda x: x[1], reverse=True)
        top_vocab = []
        count = 0
        for s in sorted_vocab_occur:
            if count < VOCAB_TOP:
                top_vocab.append(s)
                count += 1
            else:
                break
        vocabulary = [t[0] for t in top_vocab]

        for token in [GO ,UNK ,PAD ,EOS ,SPLIT] + SYMBOLS:
            if token not in vocabulary:
                vocabulary.append(token)

    if DEBUG:
        print(len(top_vocab))
        for entry in top_vocab[:10]:
            v = entry[0]
            print(v,vocab_occur[v])

    vocab_lookup = {}
    for i in range(len(vocabulary)):
        word = vocabulary[i]
        vocab_lookup[word] = i

    print("creating conversation objects...")
    with open(STRUCTURE_PATH,"r") as f:
        struct_raw = f.read().split("\n")
    for struct in struct_raw:
        if struct != "":
            subject1,subject2,movie_id,line_indices_raw = struct.split(SPACER)
            # line_indices_raw is a string of form
            # "['a','b',c']"
            # so we convert it to a list for manipulation
            line_indices = ast.literal_eval(line_indices_raw)
            conv_lines = [lines[l.strip()] for l in line_indices]
            convs.append(Conversation(subject1,subject2,conv_lines))
    seqs = []
    for j in range(len(convs)):
        print(j," of ", len(convs))
        c = convs[j]
        for i in range(len(c.lines)): # Iterate to penultimate entry because we reference next line below
            # Create entry for input_seq
            line = c.lines[i]
            encoded_line = []
            cleaned_line = preTokenize(line)
            for word in cleaned_line.split(" "):
                #encoded_line.append(word)
                if word in vocabulary:
                    encoded_line.append(vocab_lookup[word])
                else:
                    encoded_line.append(vocab_lookup[UNK])
            seqs.append(encoded_line)

    print("sequences created")
    input_seq = seqs[:-1]
    target_seq = seqs[1:]

    if DEBUG:
        #for v in vocabulary:
        #    print(v)
        print(len(vocabulary))

        c = convs[0]
        print(c.lines)

        rare = []
        for k,v in vocab_occur.items():
            if v <= 1:
                rare.append(k)
        print(len(rare))

    return input_seq,target_seq,convs,vocabulary

if __name__=="__main__":
    parseDialog()
