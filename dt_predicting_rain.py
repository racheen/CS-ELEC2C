import numpy as np
from time import time
from weather_preprocess import preprocess

def dt():
    
    features_train, features_test, labels_train, labels_test = preprocess()

    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    t0 = time()
    clf.fit(features_train,labels_train)
    print ("training time:", round(time()-t0, 3), "s")
    t1 = time()
    pred = clf.predict(features_test)
    print ("predicting time:", round(time()-t1, 3), "s")

    print (pred)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred, labels_test, normalize = True)
    #score = clf.score(features_test,labels_test)
    print("accuracy:", accuracy)
    #print ("accuracy:", score)

    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, fscore, support = precision_recall_fscore_support(labels_test, pred, average='macro')
    print ("precision:", precision)
    print ("recall:", recall)
    print ("fscore:", fscore)

    #print clf.predict_proba(features_test)
