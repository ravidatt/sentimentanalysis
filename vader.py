from nltk import *
from nltk.sentiment import vader

class VaderRuleBasedSentiments:

    '''
        This is vader based rule based sentiment analysis for 10,000 movie's review data set
        both positive and negative sentiment analysis is done in this class
    '''

    def __init__(self):
        self.sia = vader.SentimentIntensityAnalyzer()

    ''' read files from disk and load reviews into a list '''
    def readSentiments(self,file):
        with open(file,'rt') as f:
            review=f.readlines()
            return review

    ''' 
        Call vader natual language API to extract 
        over all sentiment for passed movie review 
    '''

    def vaderSentiment(self,review):
        return self.sia.polarity_scores(review)['compound']

vrbs=VaderRuleBasedSentiments()
positiveReview=vrbs.readSentiments("rt-polarity.pos")
negativeReview=vrbs.readSentiments("rt-polarity.neg")

positivePolarity=list(map(vrbs.vaderSentiment,[sent for sent in positiveReview]))
negativePolarity=list(map(vrbs.vaderSentiment,[sent for sent in negativeReview]))

percent_positive_polarity=float(sum(p>0 for p in positivePolarity))/len(positivePolarity)
percent_negative_polarity=float(sum(p>0 for p in negativePolarity))/len(negativePolarity)

print("Accuracy on positive reviews=%.2f"%(percent_positive_polarity*100)+"%")
print("Accuracy on negative reviews=%.2f"%(percent_negative_polarity*100)+"%")

