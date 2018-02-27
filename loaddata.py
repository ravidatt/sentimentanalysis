class SentimentReader:

    def  __init__(self):
        print('Load Movie Review Files For Sentiment Reader')

    ''' read files from disk and load reviews into a list '''
    def readSentiments(self, file):
        with open(file, 'rt') as f:
            review = f.readlines()
            return review