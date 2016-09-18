from heapq import nsmallest
from Amazon_review import Amazon_review
from sys import stdout
from random import shuffle
import datetime
from datetime import datetime as dt
import json
import math

training_set = []
test_set = []
small_train = 'data/sm_train.dat'
small_test = 'data/sm_test.dat'
train = 'tokenized_training_data'
test = 'tokenized_test_data'
train_data = train
test_data = test

# ----for normalization---#
begin_time = dt.now().now()


# read in all training data and make objects first
# Randomize training data
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


def find_max_min(a_list):
    min_senti = min(a_list, key=lambda review: review.mapping[0]).mapping[0]
    max_senti = max(a_list, key=lambda review: review.mapping[0]).mapping[0]
    min_learned = min(a_list, key=lambda review: review.mapping[1]).mapping[1]
    max_learned = max(a_list, key=lambda review: review.mapping[1]).mapping[1]
    min_len = min(a_list, key=lambda review: review.mapping[1]).mapping[2]
    max_len = max(a_list, key=lambda review: review.mapping[1]).mapping[2]
    return min_senti, max_senti, min_learned, max_learned, min_len, max_len


def k_NN(k, training_set, test_set):
    # for each test entity z
    # select k nearest xi's
    # call set selection algorithm
    # add z to classification set
    # print("------> In K-NN")
    neighbors = []
    i = 0
    for review in test_set:
        neighbors = nsmallest(k, training_set, key=lambda train_review: get_dist(train_review, review))
        review.test_sentiment = weighted_vote(neighbors, review)
        i += 1
        stdout.write("\r%d" % i)
        stdout.flush()
    stdout.write('\n')
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
    # pythagorean theorem
    distance = math.sqrt((x2_s - x1_s) ** 2 + (x2_l - x1_l) ** 2 + (x2_d - x1_d) ** 2)
    return distance


def weighted_vote(k_neighbors, test_review):
    # print("------> In Vote")
    # return classification vote
    # can do majority wins or distance weighted
    positive_count = 0.0
    negative_count = 0.0
    for neighbor in k_neighbors:
        dist = get_dist(test_review, neighbor)
        print(dist)
        if (neighbor.sentiment == '+1'):  # need to validate this check
            positive_count += 1 / dist

        else:
            negative_count += 1 / dist
    if (positive_count > negative_count):
        return '+1'
    else:
        return '-1'


def calculate_prediction_error(cross_validated_set):
    correct_predictions = 0.0
    num_of_predictions = len(cross_validated_set)
    for review in cross_validated_set:
        if (review.test_sentiment == review.sentiment):
            correct_predictions += 1
    # ratio of correct predictions e.g. 0.66
    error_percent = ((num_of_predictions - correct_predictions) / num_of_predictions)
    return error_percent


def get_partition_indices(current_partition, partition_size, number_of_partitions):
    left_index = int(round(partition_size * current_partition))
    right_index = int(round(partition_size * (number_of_partitions - (number_of_partitions - current_partition - 1))))
    return left_index, right_index


def get_cross_validation_lists(train_set, li, ri):
    # getting the left and right of the trainings
    if (li == 0):
        li = 1
    left = train_set[:li - 1]  # was originally li-1 but first case caused overlap
    right = []
    if (ri < len(train_set) - 1):
        right = train_set[ri:]
    tr_p = left + right
    te_p = train_set[li:ri - 1]
    # print("left index: {0} right index: {1}".format(li, ri))
    # print(tr_p[0].text)
    # print(te_p[0].text)
    # print(right[0].text)
    return tr_p, te_p


# Create Objects
sentiWord_d = read_in_dict('cleaned_senti')
trained_d = read_in_dict('training_sent_dict')
freq_d = read_in_dict('training_frequency_dict')
training_set = parse_training_set('tokenized_training_data', sentiWord_d, trained_d, freq_d)
min_sentiWord, max_sentiWord, min_learnedWord, max_learnedWord, min_length, max_length = find_max_min(training_set)

print('min/max sentiWord learnedWord, length, {0}/{1} {2}/{3} {4}/{5} '.format(min_sentiWord, max_sentiWord,
                                                                               min_learnedWord, max_learnedWord,
                                                                               min_length, max_length))
shuffle(training_set)
# Using cross validation to find parameters and attributes
# Need to find correct k (odd), attributes (word dict, length, dot product,
# etc.), attribute weights, compare object to each other
k = 5
final_k = 100
weights = {'given_sentiment': 0.40, 'training_sentiment': 0.2, 'len': 0.2, 'dot_prod': 0.2}

# breaking up the data into groups
partitions = 5
file_size = len(training_set)
partition_size = (file_size / partitions)
predictions = []
error_rate = {}
time = dt.now()
validation_file = "cross_validation_results" + str(time.hour) + str(time.minute) + '.csv'
# need to change for different attributes, k easiest right now
for k in range(final_k):
    cross_validated_test_set = []
    k += 2
    for i in range(partitions):
        # splitting into test/train partitions
        left_i, right_i = get_partition_indices(i, partition_size, partitions)
        train_partition, test_partition = get_cross_validation_lists(training_set, left_i, right_i)
        print(train_partition[:10])
        print(train_partition[0].text)
        print(test_partition[:10])
        print(test_partition[0].text)

        # predicting their sentiment
        test_predictions = k_NN(k, train_partition, test_partition)
        cross_validated_test_set = cross_validated_test_set + test_predictions

    error_rate[k] = calculate_prediction_error(cross_validated_test_set)
    time = dt.datetime.now()
    print("on {0} and the time is: {1}".format(k, time))

with open(validation_file, 'a') as csv:
    for key, value in sorted(error_rate.items()):
        csv.write('{0}\t{1}\n'.format(key, value))
