#GENERATING LABELS

import eng_spacysentiment
import pandas as pd

model = eng_spacysentiment.load()

dataset = pd.read_csv('/Users/sankeerthana/Documents/NTU/YEAR_4/SEM_2/CZ4034/IR-Sentiment-Analysis/IR-Sentiment-Analysis/skincare_dataset/crawled_reviews_5k.csv')

#removing the unnamed column
dataset.drop('Unnamed: 0', axis=1, inplace=True)

#row[2] - review title, row[3] - review 
labels_list = []

for row in dataset.itertuples():
    output = model(row[3].strip())
    max_prob = max(output.cats['positive'], output.cats['negative'])

    label = list(output.cats.keys())[list(output.cats.values()).index(max_prob)]
    
    if label == 'positive':
        labels_list.append(1)
    elif label == 'negative':
        labels_list.append(0)

#Number of Positive and Negative Labels
print(f"Positive: {labels_list.count(0)}")  #3589
print(f"Negative: {labels_list.count(1)}")  #1411 

"""
There is an imbalance in the dataset with more positive labels than negative labels.
"""
dataset['spacy_labels'] = labels_list

dataset.to_csv('/Users/sankeerthana/Documents/NTU/YEAR_4/SEM_2/CZ4034/IR-Sentiment-Analysis/IR-Sentiment-Analysis/skincare_dataset/crawled_reviews_5k_spacy.csv')


"""
result = model(text)

print(result.cats)"""