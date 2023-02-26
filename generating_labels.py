#GENERATING THE LABELS - USING TEXTBLOB

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import pandas as pd

#Reading the Dataset
dataset = pd.read_csv('/Users/sankeerthana/Documents/NTU/YEAR_4/SEM_2/CZ4034/IR-Sentiment-Analysis/IR-Sentiment-Analysis/skincare_dataset/productReviews.csv')

#dropping the unnamed column
dataset.drop(columns='Unnamed: 0', inplace=True)

#VADER Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
vader_labels = []
blob_labels = []
ctr = 0
for index, row in dataset.iterrows():
   vader_output = analyzer.polarity_scores(row['review'])
   blob = TextBlob(row['review'], analyzer=NaiveBayesAnalyzer())
   blob_pos = blob.sentiment[1]

   pos = vader_output['pos']
   neg = vader_output['neg']
   neu = vader_output['neu']
   compound = vader_output['compound']

   max_val = max(pos, neg, neu)

   idx = list(vader_output.values()).index(max_val)
   vader_sentiment = list(vader_output.keys())[idx]
  
   if vader_sentiment == 'pos':
      vader_labels.append(1)
   elif vader_sentiment == 'neg':
      vader_labels.append(-1)
   elif vader_sentiment == 'neu':
      vader_labels.append(0)

   if blob_pos < 0.50:
      blob_labels.append(-1)
   elif 0.50 < blob_pos < 0.60:
      blob_labels.append(0)
   elif blob_pos >= 0.60:
      blob_labels.append(1) 

   print(f"Completed {index}")

#convert the list to dataframe
dataset['vader_label'] = vader_labels
dataset['blob_label'] = blob_labels

dataset.to_csv('/Users/sankeerthana/Documents/NTU/YEAR_4/SEM_2/CZ4034/IR-Sentiment-Analysis/IR-Sentiment-Analysis/skincare_dataset/product_reviews_labels.csv')
    


