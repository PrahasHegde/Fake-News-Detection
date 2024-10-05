#imports
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 

#Read the data
df = pd.read_csv('news.csv')

#Get shape and head
print(df.shape)
print(df.head())


#Get the labels
labels = df.label
print(labels.head())

#Split the dataset
x_train,x_test,y_train,y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

#Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)


#Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


#Build confusion matrix
cfm = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print(cfm)
#graph
plt.figure(figsize=(20, 10))
sns.heatmap(cfm, annot=True,  fmt='d', cmap='summer')
plt.show()



""" TfidfVectorizer ->converts a collection of raw documents into a matrix of TF-IDF features."""

""" Passive Aggressive algorithms -> are online learning algorithms. Such an algorithm remains passive for a
#correct classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting.
# Unlike most other algorithms, it does not converge. Its purpose is to make updates that correct the loss, 
#causing very little change in the norm of the weight vector."""