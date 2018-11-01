import numpy as np
from time import time
from weather_preprocess import preprocess
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score

def hybrid():
    features_train, features_test, labels_train, labels_test = preprocess()    

    clf1 = tree.DecisionTreeClassifier()
    t0 = time()
    clf1.fit(features_train,labels_train)
    print ("training time:", round(time()-t0, 3), "s")
    t1 = time()
    pred = clf1.predict(features_test)
    print ("predicting time:", round(time()-t1, 3), "s")
    proba1 = clf1.predict_proba(features_train)
    proba2 = clf1.predict_proba(features_test)

    clf2 = GaussianNB()
    t0 = time()
    clf2.fit(proba1,labels_train)
    print ("training time:", round(time()-t0, 3), "s")
    t1 = time()
    pred = clf2.predict(proba2)
    print ("predicting time:", round(time()-t1, 3), "s")

    print (pred)
    accuracy = accuracy_score(pred, labels_test, normalize = True)
    #score = clf.score(features_test,labels_test)
    print("accuracy:", accuracy)
    #print ("accuracy:", score)
    precision, recall, fscore, support = precision_recall_fscore_support(labels_test, pred, average='macro')
    print ("precision:", precision)
    print ("recall:", recall)
    print ("fscore:", fscore)
