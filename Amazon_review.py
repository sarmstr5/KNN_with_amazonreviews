class Amazon_review:
    global sentiment
    global text
    global mapping 
    
    def __init__(self, text):
        self.text = text
        sentiment = ''
        mapping = 0 

    
    def toPrint(self):
        print(sentiment, text, mapping)
