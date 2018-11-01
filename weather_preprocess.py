import pandas
from sklearn import cross_validation
import numpy as np
from numpy import genfromtxt
from sklearn.cross_validation import KFold

def preprocess(weather_data = "weather.csv", rain_data = "rain.csv"):
    
    weather = pandas.read_csv(weather_data)
    rain = pandas.read_csv(rain_data)
    
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(weather, rain, test_size=0.33, random_state=42)
    return features_train, features_test, labels_train, labels_test
