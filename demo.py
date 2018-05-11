
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score as accscr
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split,cross_val_score

#Read Data from text file
df = pd.read_csv('LabelledData (1).txt',sep = '?')
#setting data frame column
df.columns = ['QUEST','CLS']
#replacing , with blank space
df['CLS'] = df['CLS'].str.replace(",","")
# filling nan values with unknown
df['CLS'] = df['CLS'].fillna('unknown')


what_df = df[df['CLS'].str.contains('what')]
who_df = df[df['CLS'].str.contains('who')]
unk_df = df[df['CLS'].str.contains('unknown')]
when_df = df[df['CLS'].str.contains('when')]
aff_df = df[df['CLS'].str.contains('affirmation')]

#joining the data 
insample = pd.concat([what_df,when_df,who_df,aff_df,unk_df],axis=0)

#setting up target and features
count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(insample['QUEST'].values)
targets = insample['CLS'].values

#spilitting data for cross validation
X_train, X_test, y_train, y_test = train_test_split(counts,targets, test_size=0.2)

#### Naive Bayes Test ####

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

clf_nb = classifier.predict(X_test)
#print clf_nb to get the output from naive bayes classifier
print 'naive bayes score ==> ',  classifier.score(X_test,y_test)


#### SVM Test ########

from sklearn import svm
model = svm.SVC(kernel='linear',gamma=1) 
model.fit(X_train,y_train)
clf_svm = model.predict(X_test)
#print clf_svm to get the output from svm classifier
print 'SVM score ==> ',  model.score(X_test,y_test)


###### RFC ############
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
clf_res = clf.predict(X_test)
#print clf_res to get the output of random forest classifier
print "RandomForestClassifier Score ==> ", clf.score(X_test,y_test)


######### LOGIT #########
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train,y_train)
clf_lr = clf.predict(X_test)
#print clf_lr to get the output of logistic regression classifier
print "LogisticRegression Score", clf.score(X_test,y_test)


