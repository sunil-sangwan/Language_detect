# python script to detect lasngugage from tweet
# you can find brute force solution for this also in file brute_force.py

# dataset is taken from https://raw.githubusercontent.com/nsorros/pyLanguage/master/data.csv

import requests
import re
# create random test and train dataset
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class lang_detector():

    def __init__(self,classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(ngram_range=(1,2),max_features=1000,preprocessor=self._remove_noise)

    # need to remove #hastage, @mention and links
    
    def _remove_noise(self, document):
        noise_pattern = re.compile("|".join(["http\S+","\@\w+","\#\w+"]))
        clean_text = re.sub(noise_pattern,"",document)
        return clean_text

    def features(self,X):
        return self.vectorizer.transform(X)
    
    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self,x):
        return self.classifier.predict(self.features([x]))

    def score(self,X,y):
        return self.classifier.score(self.features(X),y)



data_link = "https://raw.githubusercontent.com/nsorros/pyLanguage/master/data.csv"
datas     = requests.get(data_link).text
dataset   = [(line[:-3],line[-2:])for line in datas.split('\n')];
X,y       = zip(*dataset)
Xtrain,XTest,yTrain,yTest = train_test_split(X,y,random_state=1)

language_detector = lang_detector()
language_detector.fit(Xtrain,yTrain)
print(language_detector.predict('Eureko I find a great way. This is  cool'))
print (language_detector.score(XTest,yTest))