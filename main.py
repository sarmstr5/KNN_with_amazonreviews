from heapq import nsmallest

def parse_training_set():
    training_file = open("sm_train.dat","r")
    reviews = dict()
    i = 0
    for line in training_file:
        split_line = line.split('\t', 1)
        sentiment = split_line[0]
        text = split_line[1]
        review = Amazon_review(text)
        review.sentiment = sentiment
        review.dist_attr = value_calc(review)
        [i] = review
        i+=1
    return reviews
    
def parse_test_set():
    test_file = open("sm_test.dat","r")
    reviews = dict()
    i = 0
    for line in test_file:
        review = Amazon_review(line)
        review.dist_attr = value_calc(review)
        reviews[i] = review
        i+=1
    return reviews
    
def calc_feature(text_dict):
    feature_calcs = []
    for key, review in test_dict.items():
        calc = value_calc(review)
        text_dict[key].dist_attr(calc)
    return(feature_calcs)

def value_calc(review):
    return len(review.dist_attr)
    

def classify(k,training_zip, test_zip):
    #for each test entity z
    #select k nearest xi's
    #call set selection algorithm
    #add z to classification set
    #[sentiment, text, calcs]
#    for calc in test_set[2]:
#        neighbors = k-NN(k,calc,training_set)
#    for test in test_zip:
    #giving k, training list, test distances
        neighbors = k_NN(k,training_zip,list(test_zip)[1])
#        test.append(vote(neighbors))
       
def k_NN(k,training_set,test_calc):
#    sorte
#    while(k>0):
    neighbors = []
    #need to return the the tuples of the training set (at least the sentiment)
    print(list(training_set)[2])
    neighbors = nsmallest(k,list(training_set)[2], key= lambda delta: abs(delta-test_calc))
#    sorted_train = training_zip.sort()
#    while(k>0):
#        neighbors.append(
    print(neighbors)

def vote(k_neighbors):
    #return classification vote
    #can do majority wins or distance weighted
    pass
       
def main():
    training_set = []
    test_set = []

    #Design variable that needs to be updated!!!
    k = 4

    #get training set
    training_dict = parse_training_set()
    print(training_dict)
    
    #calculate features
    training_set = calc_feature(training_dict)

#    #get test set
#    #compute distance for (xi,z)
    test_dict = parse_test_set()
    test_set = calc_feature(test_dict)

    classify(k, training_set, test_set)


if __name__ == '__main__':
    main()
