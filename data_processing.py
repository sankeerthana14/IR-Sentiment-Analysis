#DATA PROCESSING

import pandas as pd
import numpy as np
from langdetect import detect
import text_processing_fxns as TEXT_PROCESSING


#Processing the dataset
dataset = pd.read_csv('/Users/sankeerthana/Documents/NTU/YEAR_4/SEM_2/CZ4034/IR-Sentiment-Analysis/IR-Sentiment-Analysis/skincare_dataset/crawled_reviews_5k_combined.csv')

#Dropping unnecessary columns
dataset.drop(['Unnamed: 0', 'Product_ID', 'Review_Title', 'spacy_labels'], axis=1, inplace=True)

for row in dataset.itertuples():
    data = row.Review

    #run the processed_pipeline
    data = TEXT_PROCESSING.processing(data)

    #remove punctuations
    data = TEXT_PROCESSING.remove_punctuation(data)
    
    #remove stopwords
    data = TEXT_PROCESSING.remove_stopwords(data)

    dataset.at[row.Index,'Review'] = ' '.join(data)

 
#Dropping an Italian sample
dataset.drop([1091], axis='index', inplace=True)

dataset.to_csv('/Users/sankeerthana/Documents/NTU/YEAR_4/SEM_2/CZ4034/IR-Sentiment-Analysis/IR-Sentiment-Analysis/skincare_dataset/processed_data_flair_labels.csv')

