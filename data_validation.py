from heapq import nsmallest
from Amazon_review import Amazon_review
from sys import stdout
from random import shuffle
from datetime import datetime as dt
#from threading import Thread
from multiprocessing.pool import ThreadPool
import json
import math
#import concurrent

training_set = []
test_set = []
small_train = 'sm_train.dat'
small_test = 'data/sm_test.dat'
train = 'tokenized_training_data'
test = 'tokenized_test_data'
train_data = small_train
test_data = test
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


def calculate_prediction_error(cross_validated_set):
    correct_predictions = 0.0
    num_of_predictions = len(cross_validated_set)
    print(num_of_predictions)
    for review in cross_validated_set:
        if (review.test_sentiment == review.sentiment):
            correct_predictions += 1
    # ratio of correct predictions e.g. 0.66
    error_percent = (correct_predictions / num_of_predictions)
    print("correct predictions: {0} the num of predictions: {1}, {2}%".format(correct_predictions, num_of_predictions,
                                                                              error_percent))
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


def data_validation(attribute, stopping_value, thread_boolean):
    pass

# Create Dictionaries
print(begin_time)
print("the file size is {0}".format(len(train_data)))
sentiWord_d = read_in_dict('cleaned_senti')
trained_d = read_in_dict('training_sent_dict')
freq_d = read_in_dict('training_frequency_dict')
training_set = parse_training_set(train, sentiWord_d, trained_d, freq_d)
min_sentiWord, max_sentiWord, min_learnedWord, max_learnedWord, min_length, max_length = find_max_min(training_set)
shuffle(training_set)

### attributes ###
k = 5
final_k =50
attribute = k
stopping_value = final_k

weights = {'given_sentiment': 0.40, 'training_sentiment': 0.2, 'len': 0.2, 'dot_prod': 0.2}
file_size = len(training_set)
partitions = 5
partition_size = (file_size / partitions)
multi_threading = False
data_validation(k, final_k, multi_threading)

# Using cross validation to find parameters and attributes
# Need to find correct k (odd), attributes (word dict, length, dot product,
# etc.), attribute weights, compare object to each other
time = dt.now()
hour = str(time.hour)
minute = str(time.minute)
if (len(minute) == 1):
    minute = '0' + minute
validation_file = "cross_validation_results" + hour + minute + '.csv'
error_rate = {}
test_predictions = []
while (attribute < stopping_value):
    cross_validated_test_set = []
    print("on {0} and the time is: {1}".format(attribute, time))
    for i in range(partitions):
        # splitting into test/train partitions
        print('this is attribute: {0} \t this is partition: {1}'.format(attribute, i))
        left_i, right_i = get_partition_indices(i, partition_size, partitions)
        train_partition, test_partition = get_cross_validation_lists(training_set, left_i, right_i)

        # predicting their sentiment
        if(multi_threading):
            pool = ThreadPool(processes=1)
            async_result = pool.apply_async(k_NN, (attribute, train_partition, test_partition))
            test_predictions = async_result.get()
        else:
            test_predictions = k_NN(attribute, train_partition, test_partition)

        print(test_predictions[0])
        cross_validated_test_set = cross_validated_test_set + test_predictions

    error_rate = calculate_prediction_error(cross_validated_test_set)
    time = dt.now()
    k += 10

    with open(validation_file, 'a') as csv:
        csv.write('{0}\t{1}\n'.format(attribute, error_rate))

##################################### Commented out code that might still be useful
# print('min/max sentiWord learnedWord, length, {0}/{1} {2}/{3} {4}/{5} '.format(min_sentiWord, max_sentiWord,
#                                                                                min_learnedWord, max_learnedWord,
#                                                                                min_length, max_length))
# executor = concurrent.futures.ProcessPoolExecutor(10)
# print(str(training_set.index(neighbor)))
# executor = concurrent.futures.ProcessPoolExecutor(4)
# futures = [executor.submit(weighted_vote, rev_and_neighbors) for review in neighbors))]
