import nltk
from loaddata import SentimentReader



    # Step 1: Split data set into training and test data sets

positiveReviewFileName="rt-polarity.pos"
negativeReviewFileName="rt-polarity.neg"
sr=SentimentReader()
positiveReviews=sr.readSentiments(positiveReviewFileName)
negativeReviews=sr.readSentiments(negativeReviewFileName)
testTrainingSplitIndex=2500
trainingNegativeReviews = negativeReviews[:testTrainingSplitIndex]
trainingPositiveReviews = positiveReviews[:testTrainingSplitIndex]
testNegativeReviews = negativeReviews[testTrainingSplitIndex+1:]
testPositiveReviews = positiveReviews[testTrainingSplitIndex+1:]

    # Step 2 Vocabulary: create vocabulary ot of training dataset

def getVocabulary():
    positiveWordList = [word for line in trainingPositiveReviews for word in line.split()]
    negativeWorldList = [word for line in trainingNegativeReviews for word in line.split()]
    allwordlist=[item for sublist in [positiveWordList,negativeWorldList] for item in sublist]
    allwordset=set(allwordlist)
    vocabulary=allwordset
    return vocabulary

    # Step 3:

def extract_features(review):
    review_words = set(review)
    features = {}
    for word in getVocabulary():
      features[word] = (word in review_words)
    return features

    # step 4

def getTrainingData():
    negTaggedTrainingReviewList =[{'review':onereview.split(),'label':'negative'} for onereview in trainingNegativeReviews]
    posTaggedTrainingReviewList=[{'review':onereview.split(),'label':'positive'} for onereview in trainingPositiveReviews]
    totallist=[sublist for mergelist in [negTaggedTrainingReviewList,posTaggedTrainingReviewList] for sublist in mergelist]
    trainingData = [(review['review'], review['label']) for review in totallist]
    return trainingData

    #step 5
def getTrainedNaiveBayesClassifier(extract_features, trainingData):
    trainingFeatures = nltk.classify.apply_features(extract_features, trainingData)
    trainedNBClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)
    return trainedNBClassifier

    # Step 6
def naiveBayesSentimentCalculator(review):
    problemInstance = review.split()
    problemFeatures = extract_features(problemInstance)
    return trainedNBClassifier.classify(problemFeatures)

vocabulary=getVocabulary()
trainingdata=getTrainingData()
trainedNBClassifier = getTrainedNaiveBayesClassifier(extract_features, trainingdata)

def getTestReviewSentiments(naiveBayesSentimentCalculator):
    testNegResults = [naiveBayesSentimentCalculator(review) for review in testNegativeReviews]
    testPosResults = [naiveBayesSentimentCalculator(review) for review in testPositiveReviews]
    labelToNum = {'positive': 1, 'negative': -1}
    numericNegResults = [labelToNum[x] for x in testNegResults]
    numericPosResults = [labelToNum[x] for x in testPosResults]
    return {'results-on-positive': numericPosResults, 'results-on-negative': numericNegResults}

def runDiagnostics(reviewResult):
  positiveReviewsResult = reviewResult['results-on-positive']
  negativeReviewsResult = reviewResult['results-on-negative']
  numTruePositive = sum(x > 0 for x in positiveReviewsResult)
  numTrueNegative = sum(x < 0 for x in negativeReviewsResult)
  pctTruePositive = float(numTruePositive)/len(positiveReviewsResult)
  pctTrueNegative = float(numTrueNegative)/len(negativeReviewsResult)
  totalAccurate = numTruePositive + numTrueNegative
  total = len(positiveReviewsResult) + len(negativeReviewsResult)
  print ("Accuracy on positive reviews = " +"%.2f" % (pctTruePositive*100) + "%")
  print ("Accurance on negative reviews = " +"%.2f" % (pctTrueNegative*100) + "%")
  print ("Overall accuracy = " + "%.2f" % (totalAccurate*100/total) + "%")



#print(nb.naiveBayesSentimentCalculator("first half awesome but 2nd half worst", trainedNBClassifier))
runDiagnostics(getTestReviewSentiments(naiveBayesSentimentCalculator))