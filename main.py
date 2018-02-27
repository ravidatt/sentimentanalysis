from vader import VaderRuleBasedSentiments
from loaddata import SentimentReader
from sentiwordnet import SentiWordNetCls
from naivebayes import NaiveBayes

class Maincls:

    sr = SentimentReader()
    positiveReview = sr.readSentiments("rt-polarity.pos")
    negativeReview = sr.readSentiments("rt-polarity.neg")

    # Vader
    vrbs=VaderRuleBasedSentiments()
    positivePolarity = list(map(vrbs.vaderSentiment, [sent for sent in positiveReview]))
    negativePolarity = list(map(vrbs.vaderSentiment, [sent for sent in negativeReview]))

    percent_positive_polarity = float(sum(p > 0 for p in positivePolarity)) / len(positivePolarity)
    percent_negative_polarity = float(sum(p > 0 for p in negativePolarity)) / len(negativePolarity)

    print("Vader: Accuracy on positive reviews=%.2f" % (percent_positive_polarity * 100) + "%")
    print("Vader: Accuracy on negative reviews=%.2f" % (percent_negative_polarity * 100) + "%")

    #SentiWord

    swnc=SentiWordNetCls()
    positivePolarity = list(map(swnc.superNaiveSentiment, [sent for sent in positiveReview]))
    negativePolarity = list(map(swnc.superNaiveSentiment, [sent for sent in negativeReview]))

    percent_positive_polarity = float(sum(p > 0 for p in positivePolarity)) / len(positivePolarity)
    percent_negative_polarity = float(sum(p > 0 for p in negativePolarity)) / len(negativePolarity)

    print("SentiWord: Accuracy on positive reviews=%.2f" % (percent_positive_polarity * 100) + "%")
    print("SentiWord: Accuracy on negative reviews=%.2f" % (percent_negative_polarity * 100) + "%")

    # Machine Learning based Naive bayes





