from heapq import nsmallest
from Amazon_review import Amazon_review
import matplotlib.pyplot as plt
import numpy as np
import csv
from nltk import corpus
from nltk import tokenize as tok

training_set = []
test_set = []
k = 4 #needs to be updated
stop_words = set(corpus.stopwords.words("english"))
stop_words.discard("not")
train_data = 'sm_train.dat'
test_data = 'sm_test.dat'
sent_dict = {}
#----for normalization---#
longest_review = 0
most_negative_review = 0
most_positive_review = 0

#------------------------#
#using sentiWordNet as dictionary for sentiment
#no particular reason to use sentiWordNet over others
def read_in_dict(word_dictionary):
    print("In read_in_dict")
    with open(word_dictionary) as csv_file:
        dict_csv = csv.reader(csv_file,delimiter = '\t')
        new_dict = {}
        for row in dict_csv:
            if row[0].startswith('#'):
                continue
            terms = row[4]
            first_term = terms.split('#')[0]
            #File has empty lines so try to read if fail skip
            #file has terms that have 0 for positive and negative weights
            #taking them out of the dict
            try:
                if float(row[2])==0 and float(row[3]) == 0.0:
                    continue
            except ValueError:
                pass
            new_dict[first_term]= [float(row[2]), float(row[3])]
        return new_dict


def plot(test_results):
    pass
    dists=[review.mapping for review in test_set]
    sentiments = [review.sentiment for review in test_set]    
    plt.plot(dists, sentiments)
    plt.show()

def tokenizer(review):
    print("------> In tokenizer\n")
    tokenized_review = tok.word_tokenize(review)
    trimmed_review = []
    for w in tokenized_review:
        if w not in stop_words:
            trimmed_review.append(w)
    return trimmed_review

def parse_training_set(training_file,sent_dict):
    print("------> In parse_training_set\n")
    reviews =[] 
    for line in training_file:
        split_line = line.split('\t', 1)
        review = Amazon_review(tokenizer(split_line[1]))
        review.sentiment = split_line[0].strip()
        review.mapping = value_calc(review, sent_dict)
        reviews.append(review)
    return reviews

def parse_test_set(test_file, sent_dict):
    print("------> In parse_test_set\n")
    reviews =[] 
    for line in test_file:
        review = Amazon_review(tokenizer(line))
        review.mapping = value_calc(review, sent_dict)
        review.sentiment = ""
        reviews.append(review)

    return reviews
    
#Need to update
def value_calc(review, sentiment_dict):
    length = len(review.text)
    pos_count = 0
    neg_count = 0
    for word in review.text:
        if word in sentiment_dict:
            #freq_dict[word]+=1
            pos_count+= sentiment_dict[word][0]
            neg_count+= sentiment_dict[word][1]

    if(most_negative_review<neg_count):
        global most_negative_review = neg_count
    if(most_positive_review<pos_count):
        global most_positive_review = pos_count
    if(longest_review<length):
        global longest_review = length

    return [pos_count, neg_count, length]
    

def k_NN(k,training_set,test_set):
    #for each test entity z
    #select k nearest xi's
    #call set selection algorithm
    #add z to classification set
    print("------> In K-NN")
    neighbors = []
    output= open("sentiment_output",'wt')
    i =0
    #need to return the the tuples of the training set (at least the sentiment)
    for review in test_set:
        neighbors = nsmallest(k,training_set, key= lambda train_review:
                              abs(train_review.mapping-review.mapping))
        review.sentiment = vote(neighbors, review)
        #print(str(test_set.index(review)),review.sentiment)
        i+=1
        print(i)
        output.write(str(review.sentiment)+'\t'+str(review.mapping)+"\n")
    return test_set

def vote(k_neighbors, test_review):
    #print("------> In Vote")
    #return classification vote
    #can do majority wins or distance weighted
    positive_count = 0
    negative_count = 0
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
    most_negative_review = 0
    most_positive_review = 0
    longest_review = 0
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
    
#    print("Training parse next feature calc")
#    training_set = calc_feature(training_list)
#    print(training_set)
#    print("test_set feature calc")
#    test_set = calc_feature(test_list)

if __name__ == '__main__':
    main()
