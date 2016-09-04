from heapq import nsmallest

def parse_training_set():
    training_file = open("sm_train.dat","r")
    sentiments = []
    texts = []
    for line in training_file:
        split_line = line.split('\t', 1)
        sentiments.append(split_line[0])
        texts.append(split_line[1])
    return sentiments, texts

def calc_feature(text_list):
    feature_calcs = []
    for line in text_list:
        calc = len(line)
        feature_calcs.append(calc)
    return(feature_calcs)

def parse_test_set():
    test_file = open("sm_test.dat","r").read()
    return test_file.split('\n')

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
    training_sentiments, training_texts = parse_training_set()

    #calculate features
    training_calcs=calc_feature(training_texts)
    #zip combines the lists together [(x1,y1,z1),(x2,y2,z2)..] use list(zip)
    training_zip = zip(training_sentiments,training_texts,training_calcs)

#    #get test set
#    #compute distance for (xi,z)
    test_text = parse_test_set()
    test_calc = calc_feature(test_text)
    test_zip = zip(test_text, test_calc)
    
    classify(k, training_zip, test_calc)


if __name__ == '__main__':
    main()
