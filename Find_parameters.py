from heapq import nsmallest
from Amazon_review import Amazon_review
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import datetime
from sys import stdout
from time import sleep
from nltk import corpus
from nltk import tokenize as tok
from random import shuffle

training_set = []
test_set = []
small_train = 'data/sm_train.dat'
small_test = 'data/sm_test.dat'
train = 'data/tokenized_training_data'
test = 'data/tokenized_test_data'
train_data = train
test_data = test

# ----for normalization---#
longest_review = 0
shortest_review = 0
max_sentiWord_review = 0
min_sentiWord_review = 0
max_positive_review = 0
min_positive_review = 0
begin_time = datetime.datetime.now().now()


# read in all training data and make objects first
# Randomize training data

def value_calc(review, sentiWord_d, learned_sent_d, freq_d):
    #    print("------> In value_calc \n")
    length = len(review.text)
    global max_sentiWord_review
    global min_sentiWord_review
    global max_positive_review
    global min_positive_review
    global longest_review
    global shortest_review
    sentiWord_sent = 0
    created_sent = 0

    for word in review.text:
        if word in sentiWord_d:
            sentiWord_sent += sentiWord_d[word][0]
        if word in learned_sent_d:
            created_sent += learned_sent_d[word][0]

    # keeping track of normalization variables
    if (max_sentiWord_review < neg_count):
        max_sentiWord_review = neg_count
    elif (min_sentiWord_review > neg_count):
        min_sentiWord_review = neg_count
    if (max_positive_review < pos_count):
        max_positive_review = pos_count
    elif (min_sentiWord_review > pos_count):
        min_sentiWord_review = pos_count
    if (longest_review < length):
        longest_review = length
    elif (shortest_review > length):
        shortest_review = length
    # if(review.max_sentiWord_review<neg_count):
    #        print(review.max_sentiWord_review)
    #        review.max_sentiWord_review=neg_count
    #        print(review.max_sentiWord_review)
    #        print (max_sentiWord_review)
    return [sentiWord_sent, created_sent, length]


def parse_training_set(training_file):
    print("------> In parse_training_set\n")
    reviews = []
    with open(training_file, 'r') as tf:
        for line in tf:
            print(line)
            split_line = line.split('\t', 1)
            review = Amazon_review(split_line[1].split(" "))
            review.sentiment = split_line[0].strip()
            review.mapping = value_calc(review, sentiWord_d, created_d, freq_d)
            reviews.append(review)
        return reviews


def find_max_min(a_list):
    min_senti = min(a_list, key=lambda review: review.mapping[0])
    max_senti = max(a_list, key=lambda review: review.mapping[0])
    min_learned = min(a_list, key=lambda review: review.mapping[1])
    max_learned = max(a_list, key=lambda review: review.mapping[1])
    min_len = min(a_list, key=lambda review: review.mapping[1])
    max_len = max(a_list, key=lambda review: review.mapping[1])
    return min_senti, max_senti, min_learned, max_learned, min_len, max_len


# Create Objects
training_set = parse_training_set(train)
min_sentiWord, max_sentiWord, min_learnedWord, max_learnedWord, min_length, max_length = find_max_min((training_set))

print('min/max sentiWord {0}/{1} \ min/max senticounter {2}/{3}'.format(min_sentiWord,max_sentiWord, max_sentiWord_review,min_sentiWord_review))

shuffle(training_set)
print(training_set)
# Using cross validation to find parameters and attributes
# Need to find correct k (odd), attributes (word dict, length, dot product,
# etc.), attribute weights, compare object to each other
k = 5
final_k = 100
weights = {}
weights['given_sentiment'] = 0.40
weights['training_sentiment'] = 0.2
weights['len'] = 0.2
weights['dot_prod'] = 0.2

# breaking up the data into groups
partitions = 5
file_size = 18192  # len(training_set)
partition_size = (file_size / partitions)
preditions = []
# need to change for different attributes, k easiest right now
for k in range(final_k):
    k += 2
    print(k)
    for i in range(partitions):
        # getting the test list
        left_i = int(round(partition_size * i))
        right_i = int(round(partition_size * (partitions - (partitions - i - 1))))
        test_list = training_set[left_i:right_i - 1]
        # getting the left and right of the test list
        left_train = training_set[:left_i - 1]
        right_train = []
        if (right_i < file_size):
            right_train = training_set[right_i:]
        train_list = left_train + right_train

        # print('[ {0}, {1} ]'.format(left_i, right_i))
