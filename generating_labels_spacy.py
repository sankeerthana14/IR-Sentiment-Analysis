#GENERATING LABELS

import eng_spacysentiment
import pandas as pd

model = eng_spacysentiment.load()

dataset = pd.read_csv('/Users/sankeerthana/Documents/NTU/YEAR_4/SEM_2/CZ4034/IR-Sentiment-Analysis/IR-Sentiment-Analysis/skincare_dataset/crawled_reviews_5k.csv')

#removing the unnamed column
dataset.drop('Unnamed: 0', axis=1, inplace=True)




"""
result = model(text)

print(result.cats)"""