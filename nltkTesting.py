import nltk.tokenize
nltk.download()

carrolTextRaw = nltk.corpus.gutenberg.raw('carroll-alice.txt')
testTextRaw = open("test.dat","r")
trainTextRaw = open("train_data","r")
testText =testTextRaw.read()
trainText = trainTextRaw.read()
print(testText)

test =nltk.tokenize.TextTilingTokenizer(demo_mode=True)
print(test)
testData = testText.encode('utf-8')
print(testData)
tokenizedData = nltk.word_tokenize(testData)

#x = test.tokenize(text2)
#x1 = test.tokenize(wholeText)
custom_sent_tokenizer = PunktSentenceTokenizer(text3)
tokenized = custom_sent_tokenizer.tokenize(sample_text)




