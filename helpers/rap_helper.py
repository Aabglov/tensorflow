# IMPORTS
import random
import os
import numpy as np # Used for One-hot encoding
import json # Used for json helper
import io
import pickle
import re

REGEX_SEARCH = '(Artist:.*\nAlbum:.*\nSong:.*\nTyped by:.*\n)'
SPACER = "||||" # Arbitrary spacer
GO = u'\xbb'
UNK = u'\xbf'
PAD = u'\xac'
EOS = u'\xf8'
SPLIT = u'\xa4'

class Vocabulary:
    def __init__(self,vocab=None,custom_go=u'\xbb' ,custom_unk=u'\xbf' ,custom_pad=u'\xf8', custom_eos=u'\xab', custom_split=u'\xac', custom_split_end=u'\xa4'):
        if not vocab:
            # DEFAULT MAGIC THE GATHERING VOCABULARY
            self.vocab = [u'\xbb','|', '5', 'c', 'r', 'e', 'a', 't', 'u', '4', '6', 'h', 'm', 'n', ' ', 'o', 'd', 'l', 'i', '7', \
                     '8', '&', '^', '/', '9', '{', 'W', '}', ',', 'T', ':', 's', 'y', 'b', 'f', 'v', 'p', '.', '3', \
                     '0', 'A', '1', 'w', 'g', '\\', 'E', '@', '+', 'R', 'C', 'x', 'B', 'G', 'O', 'k', '"', 'N', 'U', \
                     "'", 'q', 'z', '-', 'Y', 'X', '*', '%', '[', '=', ']', '~', 'j', 'Q', 'L', 'S', 'P', '2',u'\xac', u'\xf8', u'\xa4']
        else:
            self.vocab = vocab

        # Set characters
        self.go_char = custom_go
        self.unk_char = custom_unk
        self.pad_char = custom_pad
        self.eos_char = custom_eos
        self.split_char = custom_split
        self.split_end_char = custom_split_end
        for c in [custom_go,custom_unk,custom_pad,custom_eos,custom_split,custom_split_end]:
            if c not in self.vocab:
                self.vocab.append(c)
        self.eos = self.vocab.index(self.eos_char)
        self.go = self.vocab.index(self.go_char)
        self.pad = self.vocab.index(self.pad_char)
        self.unk = self.vocab.index(self.unk_char)
        self.split = self.vocab.index(self.split_char)
        self.split_end = self.vocab.index(self.split_end_char)

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

class RapParser:
    def __init__(self):
        self.REGEX_SEARCH = '(Artist:.*\nAlbum:.*\nSong:.*\nTyped by:.*\n)'
        self.SPACER = SPACER
        self.GO = GO
        self.UNK = UNK
        self.PAD = PAD
        self.EOS = EOS
        self.SPLIT = SPLIT

        #DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        self.DIR_PATH = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1]) # go up one directory
        self.DATA_PATH = os.path.join(self.DIR_PATH,"data","rap")
        self.SAVE_PATH = os.path.join(self.DATA_PATH,"rap")
        self.DEBUG = False
        #self.DEBUG = True

        self.CUSTOM_CHARS = [self.GO, self.UNK, self.PAD, self.EOS, self.SPLIT]

        self.vocab = "1 2 3 4 5 6 7 8 9 0".split(" ") + \
                     "a b c d e f g h i j k l m n o p q r s t u v w x y z".split(" ") + \
                     "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split(" ") + \
                     ['#','_','(',')', '!', '>','<'] + \
                     [' ', '&', '^', '/', '{', '}', ',', ':', '.', '\\', '@', '+'] + \
                     ['"', "'", '-', '*', '%', '[', '=', ']', '~','?',';','$'] + \
                     self.CUSTOM_CHARS

        self.foreign = ['벼', '장', '럼', '녀', '바', '닝', '빨', '감', '랑', '웃', '점', '알', '연', '남', '슛', '종', '울', '란', '음', '작', '곡', '그', '시', '론', '얼', '악', '추', '공', '경', '두', '걔', '돈', '한', '누', '짝', '팅', '딱', '침', '절', '덜', '빠', '카', '놈', '맨', '멜', '를', '속', '항', '슴', '압', '반', '협', '걱', '주', '몰', '귀', '윤', '비', '리', '낼', '최', '단', '온', '님', '새', '취', '쯤', '딜', '일', '앨', '끝', '이', '순', '켜', '었', '디', '것', '파', '명', '발', '넘', '퍼', '광', '겠', '져', '애', '포', '너', '벌', '곳', '선', '밖', '니', '워', '숙', '섭', '릴', '네', '잊', '야', '화', '않', '놓', '까', '자', '당', '피', '식', '난', '스', '히', '현', '후', '첩', '행', '찾', '플', '아', '죽', '려', '잡', '옳', '겁', '듯', '건', '여', '든', '더', '목', '덧', '걸', '했', '보', '책', '처', '내', '금', '닌', '수', '질', '지', '씩', '농', '망', '도', '뚫', '또', '된', '버', '다', '견', '담', '차', '갔', '넌', '찝', '잘', '천', '좀', '얻', '았', '왔', '티', '사', '빼', '묵', '불', '소', '박', '같', '랩', '은', '좋', '믿', '트', '낌', '닿', '쓰', '렇', '신', '킬', '왜', '릿', '편', '열', '들', '메', '싸', '멀', '줘', '년', '산', '큼', '완', '마', '터', '축', '쳐', '개', '적', '눌', '안', '볼', '셔', '회', '몸', '긴', '고', '러', '부', '실', '인', '만', '똥', '영', '젝', '에', '깨', '게', '존', '과', '와', '면', '얘', '으', '꺼', 'ñ', '상', '배', '채', '못', '름', '효', '움', '큰', '페', '짜', '왼', '친', '깝', '줄', '기', '딴', '간', '중', '의', '베', '근', '직', '뱃', '습', '따', '둘', '활', '될', '구', '득', '널', '모', '락', '래', '할', '겨', '팔', '른', '칙', '라', '능', '뿐', '커', '굴', '끼', '로', '크', '무', '석', '전', '꾸', '집', '꿈', '짓', '람', '손', '치', '린', '돼', '각', '십', '살', '하', '젠', '있', '늘', '번', '진', '확', '꾼', '가', '유', '을', '노', '범', '꼴', '우', '저', '독', '춤', '궈', '심', '달', '타', '렀', '류', '총', '프', '돌', '앞', '막', '솔', '나', '방', '폭', '통', '말', '조', '르', '숨', '서', '함', '벽', '성', '퀄', '짤', '제', '엔', '운', '복', '테', '환', '법', '써', '—', '뜨', '먹', '느', '흔', '몽', '규', '외', '위', '임', '던', '어', '였', '언', '퉁', '정', '많', '키', '욕', '힘', '생', '데', '싶', '길', '물', '레', '험', '는', '오', '씹', '력', '세', '갈', '때', '둥', '맛', '딘', '드', '거', '떨', '계', '없', '대', '동', '국', '맞', '학', '머', '냈', '놀', '원', '올', '날', '탄', '재', '문', '팬', '먼', '패', '술', '봐', '설', '매', '낸', '–', '예', '족', '홀', '해', '착', '미', '표', '북', '꼭', '송', '쾌']

        self.unsure = ['ò','ü','ç','Ç','ā', 'ï','é','à','À','è','á','¡','â','ë','î','ì']

        # These are a bunch of characters that need to removed.
        # Most of them are from unicode decoding errors
        # or belong to specific operating systems
        # so we replace them with their closest matching
        # character in ASCII
        self.REMOVE = ""
        self.SPACE = " "
        self.DOUBLE_QUOTE = '"'
        self.SINGLE_QUOTE = "'"
        self.NEWLINE = "\n"
        self.replace = {'\ufeff':self.REMOVE,
                   '\x00':self.REMOVE,
                   '\u200b':self.REMOVE,
                   '\xa0': self.SPACE,
                   '\t': self.SPACE,
                   '\x93': self.DOUBLE_QUOTE,
                   '\x94': self.DOUBLE_QUOTE,
                   '\u2028': self.NEWLINE,
                   '‘': self.SINGLE_QUOTE,
                   '“': self.DOUBLE_QUOTE,
                   '”': self.DOUBLE_QUOTE,
                   '`': self.SINGLE_QUOTE,
                   '’': self.SINGLE_QUOTE,
                   'ü','u',
                   '…': "..."
                   }

    # The plan is to separate the rap lyrics into songs,
    # then split them by line in training
    # and feed them in character by character
    def getSongs(self):
        with open(os.path.join(self.DATA_PATH,"ohhla.txt"),"r") as f:
            raw = f.read()

        for k,v in self.replace.items():
            raw = raw.replace(k,v)

        # TESTING TO MAKE SURE MY HARD CODED VOCABULARY IS CORRECT
        if self.DEBUG:
            vocab_test = list(set([r for r in raw] + self.CUSTOM_CHARS))
            print("VOCAB TEST:")
            full = self.vocab + self.unsure + self.foreign + ["\n"]
            print(set(full) == set(vocab_test))
            print(len(vocab_test))
            print(len(full))
            for f in full:
                if f not in vocab_test:
                    print("NOT IN VOCAB_TEST: |{}|".format(f))
            for v in vocab_test:
                if v not in full:
                    print("NOT IN FULL: |{}|".format(v))
            for f in full:
                if full.count(f) > 1:
                    print("COUNT: |{}|".format(f))

        sub_raw = re.sub(self.REGEX_SEARCH, self.SPACER, raw)
        songs_raw = sub_raw.split(self.SPACER)
        songs = [s.strip() for s in songs_raw if s.strip() != ""]
        print("Number of songs: {}".format(len(songs)))
        domestic_songs = []
        unsure_songs = []
        foreign_songs = []
        for s in songs:
            if set(s).intersection(self.foreign) == set() and set(s).intersection(self.unsure) == set():
                domestic_songs.append(s)
            elif set(s).intersection(self.foreign) == set() and set(s).intersection(self.unsure) != set():
                unsure_songs.append(s)
            else:
                foreign_songs.append(s)
        print("Number of songs WITHOUT non-english symbols: {}".format(len(domestic_songs)))
        print("Number of songs WITH unsure symbols: {}".format(len(unsure_songs)))
        print("Number of songs WITH non-english symbols: {}".format(len(foreign_songs)))
        return songs,self.vocab

class SongBatcher:
    def __init__(self,songs,vocab,max_seq_len=100):
        # Set up vocabulary
        self.vocab = vocab
        # Add GO and EOS char to each song
        mod_songs = []
        for s in songs:
            m = s.split("\n")
            m = [l for l in m if len(l.strip()) > 0]
            # A brief explanation:
            # GO char indicates a song is Beginning
            # split char indicates a new LINE is beginning
            # split end char indicates a LINE is ending
            # EOS char indicates a SONG is ending
            m[0] = self.vocab.go_char + m[0] + self.vocab.split_end_char
            m[-1] = self.vocab.split_char + m[-1] + self.vocab.eos_char
            m = [m[0]] + [self.vocab.split_char+l+self.vocab.split_end_char for l in m[1:-1]] + [m[-1]] # Start from 1 to skip our GO char
            mod_songs.append(m)
        self.songs = mod_songs

        self.num_batches = sum([len(s) for s in songs])

        # Set up batch generator
        self.song_index = 0
        self.num_songs = len(self.songs)
        # index within song
        self.batch_index = 0

        # Length investigation
        # Looks like there are only about 1000 songs (out of 19k)
        # that contain batches 100 words long or longer
        self.REMOVE_BIG_SEQ = True
        if self.REMOVE_BIG_SEQ:
            len_dict = {}
            songs_to_remove = []
            for i in range(len(self.songs)):
                s = self.songs[i]
                for line in s:
                    l = len(line)
                    if l not in len_dict:
                        len_dict[l] = [line]
                    else:
                        len_dict[l].append(line)
                    if l > max_seq_len:
                        songs_to_remove.append(i)
            k = list(len_dict.keys())
            k.sort()
            # total = 0
            # for i in [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 127, 128, 129, 130, 131, 133, 134, 136, 140, 142, 143, 144, 146, 147, 148, 150, 151, 152, 153, 154, 161, 162, 163, 170, 171, 172, 173, 180, 187, 198, 202, 203, 209, 213, 215, 238, 245, 249, 264, 312, 316, 565]:
            #     total += len(len_dict[i])
            # print("TOTAL: {}".format(total))
            songs_to_remove = list(set(songs_to_remove))
            print("Number of songs to remove: {}".format(len(songs_to_remove)))
            print("Number of songs before removal: {}".format(len(self.songs)))
            songs_to_remove.sort()
            for i in songs_to_remove[::-1]:
                del self.songs[i]
            print("Number of songs after removal: {}".format(len(self.songs)))

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


# This could be cleaned up into a generator function,
# but I want to be able to specifically iterate through
# values infinitely (never hit a StopIteration) so
# a classic iterator seems more appropriate
class SongSequencer:
    def __init__(self,songs,vocab,MAX_SEQ_LEN=50):
        self.song_index = 0
        self.line_index = 0
        self.num_songs = len(songs)
        self.songs = songs
        self.cur_song = self.songs[self.song_index]
        self.max_seq_len = MAX_SEQ_LEN
        self.pad_char = PAD
        self.vocab = vocab
        self.vocab_lookup = {}
        self.reverse_vocab_lookup = {}
        for i in range(len(vocab)):
            char = vocab[i]
            self.vocab_lookup[char] = i
            self.reverse_vocab_lookup[i] = char

    def padSequence(self,seq,pad_end=False):
        if len(seq) >= self.max_seq_len:
            return seq[:self.max_seq_len]
        else:
            pad_num = self.max_seq_len - len(seq)
            padding = "".join([self.pad_char] * pad_num)
            if pad_end:
                return seq + padding
            else:
                return padding + seq

    def arrayify(self,seq):
        return np.asarray([[self.vocab_lookup[s] for s in seq]])

    # Return current AND next song line
    def __next__(self):
        try:
            index_test = self.cur_song[self.line_index+1]
        except IndexError:
            self.song_index = (self.song_index + 1) % self.num_songs
            self.cur_song = self.songs[self.song_index]
            self.line_index = 0

        current_line = self.cur_song[self.line_index]
        next_line = self.cur_song[self.line_index + 1]

        ret =  self.arrayify(self.padSequence(current_line)),\
               self.arrayify(self.padSequence(next_line[:-1])), \
               self.arrayify(self.padSequence(next_line[1:],pad_end=True))

        self.line_index += 1

        return ret



def getRapData(path,max_seq_len=100):
    try:
        with open(path,"rb") as f:
            SB = pickle.load(f)
    except Exception as e:
        print(e)
        RP = RapParser()
        songs,parsed_vocab = RP.getSongs()

        vocab = Vocabulary(parsed_vocab)

        SB = SongBatcher(songs,vocab,max_seq_len)

        # Save our Rap Helper
        with open(path,"wb+") as f:
            pickle.dump(SB,f)
    return SB
