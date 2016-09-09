from nltk import corpus as corp
from nltk.tokenize import word_tokenize

example_sent = "this is an example showing how ff stop works"
stop_words = set(corp.stopwords.words("english"))
stop_words.discard("not")

words = word_tokenize(example_sent)
filtered_sentence = []

for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)

print(filtered_sentence)
