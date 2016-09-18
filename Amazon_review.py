class Amazon_review:
    global sentiment
    global text
    global mapping 
    _some_big_number = 100000000
    _least_negative_review = _some_big_number 
    _least_positive_review = _some_big_number
    _most_positive_review = 0
    _most_negative_review = 0
    _longest_review = 0
    _shortest_review = _some_big_number

    def __init__(self, text):
        self.text = text
        self.sentiment = ''
        self.mapping = 0 

    def toPrint(self):
        print(sentiment, text, mapping)

    #defining class properties for normalizing
    @property
    def least_negative_review(self):
        return self._least_negative_review
    @property
    def least_positive_review(self):
        return self._least_positive_review
    @property
    def most_negative_review(self):
        return self._most_negative_review
    @property
    def most_positive_review(self):
        return self._most_positive_review
    @property
    def longest_review(self):
        return self._longest_review
    @property
    def shortest_review(self):
        return self._shortest_review

    #creating class property setters
    @least_negative_review.setter
    def least_negative_review(self, value):
        self.least_negative_review = value
    @least_positive_review.setter
    def least_positive_review(self, value):
        self._least_positive_review = value
    @most_negative_review.setter
    def most_negative_review(self, value):
        self._most_negative_review = value
    @most_positive_review.setter
    def most_positive_review(self, value):
        self._most_positive_review = value
    @longest_review.setter
    def longest_review(self, value):
        self._longest_review = value
    @shortest_review.setter
    def shortest_review(self, value):
        self._shortest_review = value
