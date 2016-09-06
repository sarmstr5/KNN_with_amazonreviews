from heapq import nsmallest
from Amazon_review import Amazon_review
import matplotlib.pyplot as plt
import numpy as np

def plot(test_results):
    dists=[review.dist_attr for review in test_set]
    sentiments = [review.sentiment for review in test_set]    
    plt.plot(dists, sentiments)
    plt.show()

def parse_training_set(training_file):
    print("------> In parse_training_set\n")
    reviews =[] 
    for line in training_file:
        split_line = line.split('\t', 1)
        review = Amazon_review(split_line[1])
        review.sentiment = split_line[0].strip()
        review.dist_attr = value_calc(review)
        reviews.append(review)

    return reviews
    
def parse_test_set(test_file):
    print("------> In parse_test_set\n")
    reviews =[] 
    for line in test_file:
        review = Amazon_review(line)
        review.dist_attr = value_calc(review)
        review.sentiment = ""
        reviews.append(review)

    return reviews
    
def calc_feature(text_dict):
    print("------> In calc_feature\n")
    feature_calcs = []
    i = 0
    while(i<len(text_dict)): 
        #print("this is key :"+str(key))
        calc = value_calc(text_dict[i])
#        print(i,calc)
        text_dict[i].dist_attr=calc
#        print("should be the same as calc: "+ str(text_dict[i].dist_attr))
#        #there is an error with this print statement
#        print(text_dict[i].sentiment, text_dict[i].dist_attr, text_dict[i].text)
        i+=1
    return(text_dict)
#Need to update
def value_calc(review):
    return len(review.text)
    
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
                              abs(train_review.dist_attr-review.dist_attr))
        review.sentiment = vote(neighbors, review)
        #print(str(test_set.index(review)),review.sentiment)
        i+=1
        print(i)
        output.write(str(review.sentiment)+'\t'+str(review.dist_attr)+"\n")
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

def main():
    global training_set
    global test_set
    training_set = []
    test_set = []
    k = 4 #needs to be updated
    training_file = open("train_data","r")
    test_file = open("test_data","r")
    #-----------------------------------------------------#

    print("before parsing")
    #get training set and calculate distance feature
    training_dict = parse_training_set(training_file)
    print("Training parse next feature calc")
    training_set = calc_feature(training_dict)
    #print(training_set)
    #get test set, compute distance for (xi,z)
    print("Training set created, Next test dict creation")
    test_dict = parse_test_set(test_file)
    print("test_set feature calc")
    test_set = calc_feature(test_dict)
    print("calculated features for test set and training set, next classify")
    test_set = k_NN(k, training_set, test_set)
    
    training_file.close()
    test_file.close()

if __name__ == '__main__':
    main()
