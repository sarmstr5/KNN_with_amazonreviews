import csv
import json
import os

#Clean up sentiWordNet dictionary and write as JSON object
#using sentiWordNet as dictionary for sentiment
#no particular reason to use sentiWordNet over others
#senti has 6 columns (pos, id, PosScore, NegScore, SynsetTerms, Gloss)
#the column SynsetTerms has same words with different endings

current_directory = os.path.dirname(__file__)
word_dictionary = 'dictionaries/senti'

def strip_nums(s):
    return ''.join([i for i in s if not i.isdigit() or '']).strip()

with open(word_dictionary) as csv_file:
    dict_csv = csv.reader(csv_file,delimiter = '\t')
    new_dict = {}
    for row in dict_csv:
        #File has empty lines so try to read if fail skip
        #file has comments and terms that have 0 for positive and negative weights
        #taking them out of the dict
        if(row[0].startswith('#')):
            continue
        try:
            pos_score = float(row[2]) 
            neg_score = float(row[3])
            obj_score = (pos_score - neg_score)
            if obj_score == 0:
                continue
            #I have a string of terms with hashtags and nums denoting which term
            terms_no_nums = strip_nums(row[4]).strip()
            terms = terms_no_nums.split('#')
            for word in terms:
                trimmed_word = word.strip()
                if(trimmed_word == '' or '_' in trimmed_word):
                    continue
                new_dict[trimmed_word] = obj_score

        except ValueError:
            pass
with open('dictionaries/cleaned_senti', 'w') as j_file:
    json.dump(new_dict,j_file, indent = 4, sort_keys = True, separators = (',',
                                                                           ': '))
