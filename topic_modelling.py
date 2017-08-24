# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 22:20:25 2017

@author: STEFFI KERAN RANI
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import NMF,LatentDirichletAllocation

dataset = fetch_20newsgroups(shuffle=True,random_state=1,remove=('headers','footers','quotes'))
documents = dataset.data
features_no =1000
topics_no = 25

def display_topics(model,feature_names,no_top_words):
    for topic_idx,topic in enumerate(model.components_):
        print("Topic ",topic_idx,":")
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words=10

tfidf_vectorizer = TfidfVectorizer(max_df=0.95,min_df=2,max_features=features_no,stop_words='english')
tfidf=tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

tf_vectorizer=CountVectorizer(max_df=0.95,min_df=2,max_features=features_no,stop_words='english')
tf=tf_vectorizer.fit_transform(documents)
tf_feature_names=tf_vectorizer.get_feature_names()

# Run NMF
nmf = NMF(n_components = topics_no,random_state=1,alpha=.1,l1_ratio=.5,init='nndsvd').fit(tfidf)

#Run LDA
lda = LatentDirichletAllocation(n_topics=topics_no,max_iter=5,learning_method='online',learning_offset=50, random_state=0).fit(tf)

display_topics(nmf,tfidf_feature_names,no_top_words=10)
display_topics(lda,tf_feature_names,no_top_words=10)
