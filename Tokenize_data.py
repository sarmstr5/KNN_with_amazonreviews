#Tokenize Test and Training files then write them as shortened file

import csv
import json
import os
from Amazon_review import Amazon_review
import csv
from nltk import corpus
from nltk import tokenize as tok
import operator

stop_words = set(corpus.stopwords.words("english"))
stop_words.discard("not")
stop_words.add("i")
training_file = 'data/train_data'
test_file = 'data/test_data'
training_list = []
test_list = []

def tokenizer(review):
#    print("------> In tokenizer\n")
    tokenized_review = tok.word_tokenize(review)
    trimmed_review = []
    for word in tokenized_review:
        if word not in stop_words:
            if '.' in word:
                new_w=word.split('.')
                for i in new_w:
                    if(i == ''):
                        continue
                    trimmed_review.append(i.lower())
                    trimmed_review.append(" ")
            else:
                trimmed_review.append(word.lower())
                trimmed_review.append(" ")

    return ''.join(trimmed_review)

def create_token_list(original_file):
    print(original_file)
    review_list = []
    with open(original_file, 'r') as tf:
        if('train' in original_file):
            for line in tf:
                split_line = line.split('\t', 1)
                review_sentiment = split_line[0]
                #list of words without stop words
                review = tokenizer(split_line[1])
                review_list.append([review_sentiment, review])
        else:
            for line in tf:
                #list of words without stop words
                review = tokenizer(line)
                review_list.append(review)
    return review_list 

def create_token_string(text_name, tokenized_list):
    with open(text_name, 'w') as tf:
        if('train' in text_name):
            for review in tokenized_list:
                tf.write(str(review[0])+'\t'+str(review[1])+'\n')
        else:
            for review in tokenized_list:
                tf.write(str(review)+'\n')

test_list = create_token_list(test_file)
training_list = create_token_list(training_file)

create_token_string('data/tokenized_training_data', training_list)
create_token_string('data/tokenized_test_data', test_list)

