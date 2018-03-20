# IMPORTS
import random
import os
import numpy as np # Used for One-hot encoding
import json # Used for json helper
import ast # USed to parse List from string ("['a','b',c']")
#import caffeine
import io

import re

REGEX_SEARCH = '(Artist:.*\nAlbum:.*\nSong:.*\nTyped by:.*\n)'
SPACER = "||||" # Arbitrary spacer
GO = u'\xbb'
UNK = u'\xbf'
PAD = u'\xac'
EOS = u'\xf8'
SPLIT = u'\xa4'

#DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_PATH = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1]) # go up one directory
DATA_PATH = os.path.join(DIR_PATH,"data","rap")
SAVE_PATH = os.path.join(DATA_PATH,"rap")
DEBUG = False
#DEBUG = True

CUSTOM_CHARS = [GO, UNK, PAD, EOS, SPLIT]

vocab = "1 2 3 4 5 6 7 8 9 0".split(" ")
vocab += "a b c d e f g h i j k l m n o p q r s t u v w x y z".split(" ")
vocab += "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split(" ")
vocab += ['#','_','(',')', '!', '>','<']
vocab += [' ', '&', '^', '/', '{', '}', ',', ':', '.', '\\', '@', '+']
vocab += ['"', "'", '-', '*', '%', '[', '=', ']', '~','?',';','$']
vocab += CUSTOM_CHARS

foreign = ['벼', '장', '럼', '녀', '바', '닝', '빨', '감', '랑', '웃', '점', '알', '연', '남', '슛', '종', '울', '란', '음', '작', '곡', '그', '시', '론', '얼', '악', '추', '공', '경', '두', '걔', '돈', '한', '누', '짝', '팅', '딱', '침', '절', '덜', '빠', '카', '놈', '맨', '멜', '를', '속', '항', '슴', '압', '반', '협', '걱', '주', '몰', '귀', '윤', '비', '리', '낼', '최', '단', '온', '님', '새', '취', '쯤', '딜', '일', '앨', '끝', '이', '순', '켜', '었', '디', '것', '파', '명', '발', '넘', '퍼', '광', '겠', '져', '애', '포', '너', '벌', '곳', '선', '밖', '니', '워', '숙', '섭', '릴', '네', '잊', '야', '화', '않', '놓', '까', '자', '당', '피', '식', '난', '스', '히', '현', '후', '첩', '행', '찾', '플', '아', '죽', '려', '잡', '옳', '겁', '듯', '건', '여', '든', '더', '목', '덧', '걸', '했', '보', '책', '처', '내', '금', '닌', '수', '질', '지', '씩', '농', '망', '도', '뚫', '또', '된', '버', '다', '견', '담', '차', '갔', '넌', '찝', '잘', '천', '좀', '얻', '았', '왔', '티', '사', '빼', '묵', '불', '소', '박', '같', '랩', '은', '좋', '믿', '트', '낌', '닿', '쓰', '렇', '신', '킬', '왜', '릿', '편', '열', '들', '메', '싸', '멀', '줘', '년', '산', '큼', '완', '마', '터', '축', '쳐', '개', '적', '눌', '안', '볼', '셔', '회', '몸', '긴', '고', '러', '부', '실', '인', '만', '똥', '영', '젝', '에', '깨', '게', '존', '과', '와', '면', '얘', '으', '꺼', 'ñ', '상', '배', '채', '못', '름', '효', '움', '큰', '페', '짜', '왼', '친', '깝', '줄', '기', '딴', '간', '중', '의', '베', '근', '직', '뱃', '습', '따', '둘', '활', '될', '구', '득', '널', '모', '락', '래', '할', '겨', '팔', '른', '칙', '라', '능', '뿐', '커', '굴', '끼', '로', '크', '무', '석', '전', '꾸', '집', '꿈', '짓', '람', '손', '치', '린', '돼', '각', '십', '살', '하', '젠', '있', '늘', '번', '진', '확', '꾼', '가', '유', '을', '노', '범', '꼴', '우', '저', '독', '춤', '궈', '심', '달', '타', '렀', '류', '총', '프', '돌', '앞', '막', '솔', '나', '방', '폭', '통', '말', '조', '르', '숨', '서', '함', '벽', '성', '퀄', '짤', '제', '엔', '운', '복', '테', '환', '법', '써', '—', '뜨', '먹', '느', '흔', '몽', '규', '외', '위', '임', '던', '어', '였', '언', '퉁', '정', '많', '키', '욕', '힘', '생', '데', '싶', '길', '물', '레', '험', '는', '오', '씹', '력', '세', '갈', '때', '둥', '맛', '딘', '드', '거', '떨', '계', '없', '대', '동', '국', '맞', '학', '머', '냈', '놀', '원', '올', '날', '탄', '재', '문', '팬', '먼', '패', '술', '봐', '설', '매', '낸', '–', '예', '족', '홀', '해', '착', '미', '표', '북', '꼭', '송', '쾌']

unsure = ['ò','ü','ç','Ç','ā', 'ï','é','à','À','è','á','¡','â','ë','î','ì']

# These are a bunch of characters that need to removed.
# Most of them are from unicode decoding errors
# or belong to specific operating systems
# so we replace them with their closest matching
# character in ASCII
REMOVE = ""
SPACE = " "
DOUBLE_QUOTE = '"'
SINGLE_QUOTE = "'"
NEWLINE = "\n"
replace = {'\ufeff':REMOVE,
           '\x00':REMOVE,
           '\u200b':REMOVE,
           '\xa0': SPACE,
           '\t': SPACE,
           '\x93': DOUBLE_QUOTE,
           '\x94': DOUBLE_QUOTE,
           '\u2028': NEWLINE,
           '‘': SINGLE_QUOTE,
           '“': DOUBLE_QUOTE,
           '”': DOUBLE_QUOTE,
           '`': SINGLE_QUOTE,
           '’': SINGLE_QUOTE,
           '…': "..."
           }

# The plan is to separate the rap lyrics into songs,
# then split them by line in training
# and feed them in character by character
def getSongs():
    with open(os.path.join(DATA_PATH,"ohhla.txt"),"r") as f:
        raw = f.read()

    for k,v in replace.items():
        raw = raw.replace(k,v)

    # TESTING TO MAKE SURE MY HARD CODED VOCABULARY IS CORRECT
    if DEBUG:
        vocab_test = list(set([r for r in raw] + CUSTOM_CHARS))
        print("VOCAB TEST:")
        full = vocab + unsure + foreign + ["\n"]
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

    sub_raw = re.sub(REGEX_SEARCH, SPACER, raw)
    songs_raw = sub_raw.split(SPACER)
    songs = [s.strip() for s in songs_raw if s.strip() != ""]
    print("Number of songs: {}".format(len(songs)))
    domestic_songs = []
    unsure_songs = []
    foreign_songs = []
    for s in songs:
        if set(s).intersection(foreign) == set() and set(s).intersection(unsure) == set():
            domestic_songs.append(s)
        elif set(s).intersection(foreign) == set() and set(s).intersection(unsure) != set():
            unsure_songs.append(s)
        else:
            foreign_songs.append(s)
    print("Number of songs WITHOUT non-english symbols: {}".format(len(domestic_songs)))
    print("Number of songs WITH unsure symbols: {}".format(len(unsure_songs)))
    print("Number of songs WITH non-english symbols: {}".format(len(foreign_songs)))
    return songs,vocab

if __name__ == "__main__":
    s,v = getSongs()
