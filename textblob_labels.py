#GENERATING THE LABELS - USING TEXTBLOB

from textblob.sentiments import NaiveBayesAnalyzer
from textblob import TextBlob
import pandas as pd

#Reading the Dataset
dataset = pd.read_csv('')
#TextBlob Sentiment

blob = TextBlob("I love this library", analyzer=NaiveBayesAnalyzer())
print(blob.sentiment)