#Creating and writing the training dictionary for classifying
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
sent_dict = {}
freq_dict = {}
training_file = 'data/train_data'

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
            else:
                trimmed_review.append(word.lower())

    return trimmed_review

with open(training_file, 'r') as training_file:
    for line in training_file:
        split_line = line.split('\t', 1)
        review_sentiment = split_line[0]
        #list of words without stop words
        review = tokenizer(split_line[1])
        for word in review:
            #if word doesnt exist in dictionary default value is 0
            freq_dict[word] = freq_dict.get(word, 0) +1
            if(review_sentiment == '+1'):
                sent_dict[word] = sent_dict.get(word,0)+1
            elif(review_sentiment == '-1'):
                sent_dict[word] = sent_dict.get(word,0)-1 
            else:
                print("review didnt have sentiment???")
                print(review)

#read in each review of the training set
#take out stop words
# add words from review to frequency dictionary as key value =+1
# add words from review to sentiment dictionary as key, value =+ sentiment from review
# only keep words with most sentiment and freq
trimmed_freq_dict = {}
trimmed_sent_dict = {}
# trimming sentiment and frequency dictionaries
# dictionaries are hashtables so turning into tuples to sort by value
freq_sorted = sorted(freq_dict.items(), key = lambda x: x[1], reverse = True)
sent_sorted = sorted(sent_dict.items(), key = lambda x: x[1])
trimmed_freq_index = int(len(freq_sorted)*.3)
trimmed_sent_index = int(len(sent_sorted)*.15)
trimmed_sent_index_from_end = len(sent_sorted)-trimmed_sent_index

#using the tuples to create trimmed dictionaries
for tup in freq_sorted[:trimmed_freq_index]:
    trimmed_freq_dict[tup[0]] = tup[1]
for tup in sent_sorted[:trimmed_sent_index]: 
    trimmed_sent_dict[tup[0]] = tup[1]
for tup in sent_sorted[trimmed_sent_index_from_end:]:
    trimmed_sent_dict[tup[0]] = tup[1]

#writing trimmed dictionaries to json files
with open('dictionaries/training_frequency_dict', 'w') as j_file:
    json.dump(trimmed_freq_dict, j_file, indent = 4, sort_keys = True, separators=(',',': '))
with open('dictionaries/training_sent_dict', 'w') as j_file:
    json.dump(trimmed_sent_dict, j_file, indent = 4, sort_keys = True, separators=(',' , ': '))

