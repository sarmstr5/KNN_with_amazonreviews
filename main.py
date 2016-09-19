import matplotlib.pyplot as plt
from heapq import nsmallest
from Amazon_review import Amazon_review
from sys import stdout
from random import shuffle
from datetime import datetime as dt
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, cpu_count
from threading import Thread
import json
import math
import concurrent

training_set = []
test_set = []
small_train = 'sm_train.dat'
small_test = 'sm_test.dat'
train = 'tokenized_training_data'
test = 'tokenized_test_data'
train_data = train
test_data = test
begin_time = dt.now().now()


# using sentiWordNet as dictionary for given sentiment
# no particular reason to use sentiWordNet over others
# cleaned and saved all dictionaries as a JSON files
def read_in_dict(word_dictionary):
    with open(word_dictionary) as js:
        sentiment_dict = json.load(js)
    return sentiment_dict


def value_calc(review, sentiWord_d, learned_sent_d, freq_d):
    #    print("------> In value_calc \n")
    length = len(review.text)
    sentiWord_sent = 0.0
    created_sent = 0.0

    for word in review.text:
        if word in sentiWord_d:
            sentiWord_sent += float(sentiWord_d[word])
        if word in learned_sent_d:
            created_sent += float(learned_sent_d[word])
    return [sentiWord_sent, created_sent, length]


def parse_training_set(training_file, sentiWord_dict, trained_dict, freq_dict):
    print("------> In parse_training_set\n")
    reviews = []
    with open(training_file, 'r') as tf:
        for line in tf:
            split_line = line.split('\t', 1)
            review = Amazon_review(split_line[1].split(" "))
            review.sentiment = split_line[0].strip()
            review.mapping = value_calc(review, sentiWord_dict, trained_dict, freq_dict)
            reviews.append(review)
        return reviews


def parse_test_set(test_file, sentiWord_dict, trained_dict, freq_dict):
    print("------> In parse_test_set\n")
    reviews = []
    with open(test_file, 'r') as tf:
        for line in tf:
            review = Amazon_review(line.split(" "))
            review.mapping = value_calc(review, sentiWord_dict, trained_dict, freq_dict)
            review.sentiment = ""
            reviews.append(review)
        return reviews


def find_max_min(a_list):
    min_senti = min(a_list, key=lambda review: review.mapping[0]).mapping[0]
    max_senti = max(a_list, key=lambda review: review.mapping[0]).mapping[0]
    min_learned = min(a_list, key=lambda review: review.mapping[1]).mapping[1]
    max_learned = max(a_list, key=lambda review: review.mapping[1]).mapping[1]
    min_len = min(a_list, key=lambda review: review.mapping[2]).mapping[2]
    max_len = max(a_list, key=lambda review: review.mapping[2]).mapping[2]
    return min_senti, max_senti, min_learned, max_learned, min_len, max_len


def k_NN(k, training_set, test_set):
    # for each test entity z
    # select k nearest xi's
    # call set selection algorithm
    # add z to classification set
    i = 0
    for review in test_set:
        neighbors = nsmallest(k, training_set, key=lambda train_review: get_dist(train_review, review))
        rev_and_neighbors = [neighbors, review]
        review.test_sentiment = weighted_vote(rev_and_neighbors)
    # i += 1
    #     stdout.write("\r%d"%i)
    #     stdout.flush()
    # stdout.write('\n')
    return test_set


def normalize(x, xmin, xmax):
    return ((x - xmin) / (xmax - xmin))


def get_dist(review1, review2):
    # assuming variables are linear, probably not true and have not tested
    x1_s = normalize(review1.mapping[0], min_sentiWord, max_sentiWord)
    x2_s = normalize(review2.mapping[0], min_sentiWord, max_sentiWord)
    x1_l = normalize(review1.mapping[1], min_learnedWord, max_learnedWord)
    x2_l = normalize(review2.mapping[1], min_learnedWord, max_learnedWord)
    x1_d = normalize(review1.mapping[2], min_length, max_length)
    x2_d = normalize(review2.mapping[2], min_length, max_length)
    # have not done anything for the dot product yet
    # weights of sentiWord, learnedWord, length
    w1, w2, w3 = .5, .35, .15
    # pythagorean theorem
    distance = math.sqrt(w1 * (x2_s - x1_s) ** 2 + w2 * (x2_l - x1_l) ** 2 + w3 * (x2_d - x1_d) ** 2)
    return distance


def weighted_vote(neighbors_and_test):
    # return classification vote
    # can do majority wins or distance weighted
    k_neighbors, test_review = neighbors_and_test[0], neighbors_and_test[1]
    positive_count = 0.0
    negative_count = 0.0
    for neighbor in k_neighbors:
        dist = get_dist(test_review, neighbor)
        if (dist == 0):
            # can test by reading out to file
            continue
        if (neighbor.sentiment == '+1'):  # need to validate this check
            positive_count += 1 / dist

        else:
            negative_count += 1 / dist
    if (positive_count > negative_count):
        return '+1'
    else:
        return '-1'


def print_results_to_csv(test_list):
    time = dt.now()
    hour, minute = str(time.hour), str(time.minute)
    if (len(minute) == 1):
        minute = '0' + minute
    if (len(hour) == 1):
        hour = '0' + hour
    test_sentiment_output = "test_results" + hour + minute + '.csv'
    test_reviews = "test_reviews" + hour + minute + '.csv'
    with open(test_sentiment_output, 'w') as results:
        for review in test_list:
            results.write('{0}\n'.format(review.test_sentiment))

    with open(test_reviews, 'w') as csv:
        r = test_list[0]
        "predicted sentiment\ttext \tsentiWord\tlearnedWord\tlength\n".format(r.test_sentiment, r.text, r.mapping[0],
                                                                              r.mapping[1], r.mapping[2])
        for review in test_list:
            # results.write(review.toPrint())
            csv.write('{0}\t{1}\n'.format(review.test_sentiment, review.text))

if __name__ == '__main__':
    # Beginning Script
    # Create Dictionaries
    print("Initializing Script")
    print(begin_time)
    sentiWord_d = read_in_dict('cleaned_senti')
    trained_d = read_in_dict('training_sent_dict')
    freq_d = read_in_dict('training_frequency_dict')

    # Read in training file
    print("Creating Test Set")
    training_list = parse_training_set(train_data, sentiWord_d, trained_d, freq_d)
    print("the training file size is {0}".format(len(training_list)))
    min_sentiWord, max_sentiWord, min_learnedWord, max_learnedWord, min_length, max_length = find_max_min(training_list)

    # Read in text file
    print("Training set and sentiment dictionaries created.  Next test predictions")
    test_list = parse_test_set(test_data, sentiWord_d, trained_d, freq_d)
    print("the test file size is {0}".format(len(test_list)))

    ### attributes ###
    print("Predicting Review Sentiments")
    k = 3
    regular_run = False
    multithreading = False*(1-regular_run)
    multiprocessing = (not multithreading)*(1-regular_run)
    # Classifying test set using multithreading or multiprocessing or neither
    # k_NN is the classifying function. Arguments of k_NN are passed as a tuple
    while (k < 100):
        t = dt.now().now()
        print("Beginning classification at\n{0}\n".format(dt.now().now()))
        if ( multithreading ):
            pool = ThreadPool(processes=2)
            async_result = pool.apply_async(k_NN, (k, training_list, test_list))
            updated_test_set = async_result.get()
        elif( multiprocessing ):
            pool = Pool(processes=cpu_count())
            updated_test_set = k_NN(k, training_list, test_list)
            async_result = pool.apply_async(k_NN, (k, training_list, test_list))
            updated_test_set = async_result.get()
        else:
            k_NN(k, training_list, test_list)

        print(dt.now().now())

        # print results as csv for upload
        print("Printing test output files\n")
        print("The size of the test set {0}, size of updated test set: {2}".format(len(updated_test_set),
                                                                                   len(test_list)))
        print_results_to_csv(updated_test_set)
        t2 = dt.now().now()
        with open('time.txt', 'a') as txt:
            txt.write("{0}\t{1}\t{2}\n".format(k, t, t2))
        k += 5

    # def plot(test_results):
    #     pass
    #     dists=[review.mapping for review in test_set]
    #     sentiments = [review.sentiment for review in test_set]
    #     plt.plot(dists, sentiments)
    #     plt.show()

