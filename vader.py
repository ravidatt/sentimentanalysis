from nltk import *
from nltk.sentiment import vader
from loaddata import SentimentReader

class VaderRuleBasedSentiments():

    '''
        This is vader based rule based sentiment analysis for 10,000 movie's review data set
        both positive and negative sentiment analysis is done in this class
    '''

    def __init__(self):
        self.sia = vader.SentimentIntensityAnalyzer()

    ''' 
        Call vader rule based natural language API to extract 
        over all sentiment for passed movie review 
    '''

    def vaderSentiment(self,review):
        return self.sia.polarity_scores(review)['compound']



