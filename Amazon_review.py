class Amazon_review:
    test_sentiment = 0

    def __init__(self, text):
        self.text = text
        self.sentiment = ''
        self.mapping = []

    def toPrint(self):
        print("\t{0}\t{1}\t{2}\t{3}\t{4}\n".format(self.test_sentiment, self.text, self.mapping[0], self.mapping[1],
                                                   self.mapping[2]))
