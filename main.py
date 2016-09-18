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

training_set = []
test_set = []
k = 5 #needs to be updated
small_train = 'data/sm_train.dat'
small_test = 'data/sm_test.dat'
train = 'data/tokenized_train_data'
test = 'data/tokenized_test_data'
train_data = train
test_data = test

#----for normalization---#
longest_review = 0
shortest_review = 0
most_negative_review = 0
least_negative_review = 0
most_positive_review = 0
least_positive_review = 0
begin_time = datetime.datetime.now().now()
#------------------------#
#using sentiWordNet as dictionary for sentiment
#no particular reason to use sentiWordNet over others
#cleaned and saved as a JSON file
def read_in_dict(word_dictionary):
    with open('dictionary/cleaned_senti') as js:
        sentiment_dict = json.load(js)
    return sentiment_dict 

def plot(test_results):
    pass
    dists=[review.mapping for review in test_set]
    sentiments = [review.sentiment for review in test_set]    
    plt.plot(dists, sentiments)
    plt.show()

def parse_training_set(training_file,sent_dict):
    print("------> In parse_training_set\n")
    reviews =[] 
    for line in training_file:
        split_line = line.split('\t', 1)
        review = Amazon_review(tok.word_tokenize(split_line[1]))
        review.sentiment = split_line[0].strip()
        review.mapping = value_calc(review, sent_dict)
        reviews.append(review)
    return reviews

def parse_test_set(test_file, sent_dict):
    print("------> In parse_test_set\n")
    reviews =[] 
    for line in test_file:
        review = Amazon_review(tok.word_tokenize(line))
        review.mapping = value_calc(review, sent_dict)
        review.sentiment = ""
        reviews.append(review)
    return reviews
# I have the given dictionary and one created from training set    
def value_calc(review, sentiWord_d, created_sent_d, freq_d):
#    print("------> In value_calc \n")
    length = len(review.text)
    global most_negative_review
    global least_negative_review
    global most_positive_review
    global least_positive_review
    global longest_review
    global shortest_review
    given_pos_count = 0
    given_neg_count = 0
    my_neg_count = 0
    my_pos_count = 0

    for word in review.text:
        if word in sentiment_dict:
            pos_count+= sentiment_dict[word][0]
            neg_count+= sentiment_dict[word][1]

    #keeping track of normalization variables
    if(most_negative_review<neg_count):
        most_negative_review = neg_count
    elif(least_negative_review>neg_count):
        least_negative_review = neg_count
    if(most_positive_review<pos_count):
        most_positive_review = pos_count
    elif(least_negative_review>pos_count):
        least_negative_review = pos_count
    if(longest_review<length):
        longest_review = length
    elif(shortest_review>length):
        shortest_review = length
#    if(review.most_negative_review<neg_count):
#        print(review.most_negative_review)
#        review.most_negative_review=neg_count
#        print(review.most_negative_review)
#        print (most_negative_review)
    return [pos_count, neg_count, length]

def k_NN(k,training_set,test_set):
    #for each test entity z
    #select k nearest xi's
    #call set selection algorithm
    #add z to classification set
    #print("------> In K-NN")
    neighbors = []
    time = datetime.datetime.now()
    output= open("sentiment_output"+str(time.hour)+str(time.minute),'wt')
    i =0
    #need to return the the tuples of the training set (at least the sentiment)
    for review in test_set:
        neighbors = nsmallest(k,training_set, key= lambda train_review:
                              get_dist(train_review, review))
        review.sentiment = vote(neighbors, review)
        #print(str(test_set.index(review)),review.sentiment)
        i+=1
        #output.write(str(review.sentiment)+'\t'+str(review.mapping)+"\n")
        output.write(str(review.sentiment)+"\n")
        stdout.write("\r%d"%i)
        stdout.flush()
    stdout.write('\n')
    return test_set

def normalize(x, xmin, xmax):
    return((x - xmin)/(xmax - xmin))

def get_dist(review1, review2):
#    global most_negative_review 
#    global least_negative_review
#    global most_positive_review
#    global least_positive_review
#    global longest_review
#    global shortest_review
    r1_p = normalize(review1.mapping[0],least_positive_review,most_positive_review)
    r1_n = normalize(review1.mapping[1],least_negative_review,most_negative_review)
    r1_d = normalize(review1.mapping[2],shortest_review, longest_review)
    r2_p = normalize(review2.mapping[0],least_positive_review,most_positive_review)
    r2_n = normalize(review2.mapping[1],least_negative_review,most_negative_review)
    r2_d = normalize(review2.mapping[2],shortest_review, longest_review)
    #pythagorean theorem d = sqrt((x2-x1)^2+(y2-y1)^2)
    distance = math.sqrt((r2_p-r1_p)**2 + (r2_n - r1_n)**2 + (r2_d - r1_d)**2)
    #print(distance)
    return distance

def vote(k_neighbors, test_review):
    #print("------> In Vote")
    #return classification vote
    #can do majority wins or distance weighted
    positive_count = 0
    negative_count = 0
    weighted_vote = 0
    for neighbor in k_neighbors:
        if(neighbor.sentiment == '+1'): #need to validate this check
            positive_count+=1
            
        else:
            negative_count+=1
    if(positive_count>negative_count): 
        return '+1'
    else:
        return '-1'
def reduce_dictionary(sentiment_dict):
    pass

def main():
    training_file = open(train_data,"r")
    test_file = open(test_data,"r")
    sent_dict = read_in_dict('senti')
    #-----------------------------------------------------#

    print("Initializing program")
    #get training set and calculate distance feature
    training_list = parse_training_set(training_file,sent_dict)

    #get test set and set attributes
    print("Training set created, Next test creation")
    test_list = parse_test_set(test_file, sent_dict)

    #reduce the dictionary to the most common words
    reduce_dictionary(sent_dict)

    #Classify test set
    test_set = k_NN(k, training_list, test_list)
    training_file.close()
    test_file.close()
    print(begin_time)
    print(datetime.datetime.now().time())
    
#    print("Training parse next feature calc")
#    training_set = calc_feature(training_list)
#    print(training_set)
#    print("test_set feature calc")
#    test_set = calc_feature(test_list)

if __name__ == '__main__':
    main()
