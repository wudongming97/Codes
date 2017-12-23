# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

T_ = {
    'to_lower': True,
    'remove_punctuations': True,
    'remove_stop_words': True,
    'need_stem': True,
    'need_lemma': True,

    'line_lens': (5, 60),
    'word_lens': (5, 25),
}

stopwords = set(stopwords.words('english'))
snowball = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

if __name__ == '__main__':
    raw_file = '../data/europarl-v7.txt'
    target_file = '../data/europarl-v7.txt_'
    with open(raw_file, 'r', encoding='utf-8') as sf:
        with open(target_file, 'w', encoding='utf-8') as tf:
            for line in sf:
                # 以行为单位处理逻辑
                if T_.get('to_lower'):
                    line = line.lower()
                if T_.get('remove_punctuations'):
                    line = line.translate(str.maketrans("", "", string.punctuation))

                tokens = nltk.word_tokenize(line)

                # 以words为单位进行处理
                if T_.get('remove_stop_words'):
                    tokens = [w for w in tokens if w not in stopwords]
                if T_.get('need_stem'):
                    tokens = [snowball.stem(w) for w in tokens]
                if T_.get('need_lemma'):
                    tokens = [lemmatizer.lemmatize(w) for w in tokens]

                if (len(tokens) < T_.get('word_lens')[0] or len(tokens) >= T_.get('word_lens')[1]):
                    continue
                line = ' '.join(tokens)
                if len(line) < T_.get('line_lens')[0] or len(line) >= T_.get('line_lens')[1]:
                    continue

                tf.write(line + '\n')
