class Amazon_review:
    global sentiment
    global text
    global dist_attr 
    
    def __init__(self, text):
        self.text = text
        sentiment = ''
        dist_attr = 0 

    
    def toPrint(self):
        print(sentiment, text, dist_attr)
