# sentiment analysis
Sentiment Analysis using Rule Based & Machine Learning based Binary Classifiers:

Sentiment analysis is a binary classification which
takes in a problem instance as input and classify into a label.

There are 2 types of classifiers : Rule based and ML based.

Rule based approach:
1) Static model- rules are applied independent of data being analysed
2) Subject matter experts required; Rules drawn up by experts.
3) Corpus optional (data)
4) No training steps
5) Can operate on isolated problem instances without large data sets

Example:

1) Vader: (Valence Aware Dictionary for Sentiment Reasoning) : VADER

    Built into python nltk packaage: Natural Language Tool Kit

it supports:
     Emoticons
     Idioms
     Punctuations
     Negation
     Emphasis
     Contrast


2) Sentiworld:
    Sentiment lexicon with polarity information
    Built into python NLTK
    Very convenient to use
    Extends WordNet

Install :
pip install nltk
nltk.download() # to download lexicon to your local hard disk

Machine Learning Based approach:

1) Dynamic- alter output based on pattern in data.
2) Subject matter expert optional
3) Corpus required (data sets to train models):
   label is assigned based on patterns displayed in aggregate data.
   Classification algorithm works behind the scene for ML based Classifiers.
4) Training steps needed
5) Can only work if data set is available

Naive bayes Classifier:
NB is supervised Machine learning method.

6 Steps process

1) Split corpus into training data and test data
2) Define Vocabulary
3) Extract features
4) Train classifier
5) Classify test data
6) Measure Accuracy



