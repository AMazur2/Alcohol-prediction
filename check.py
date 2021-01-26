import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import math

from RandomForest import RandomForest


def loadData():
    features = pd.read_csv('temps.csv')
    features=features.drop(["forecast_noaa","forecast_acc","forecast_under"],axis=1)
    features = pd.get_dummies(features)

    # Labels are the values we want to predict
    labels = np.array(features['actual'])
    # print(features['actual'].head())
    # print(labels.shape)
    # print(labels)

    # Remove the labels from the features
    features = features.drop('actual', axis=1)
    # Saving feature names for later use
    feature_list = list(features.columns)
    features = np.array(features)
    # print(features)
    return features, labels, feature_list

def splitData(features, labels):
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=42)

    # print('Training Features Shape:', train_features.shape)
    # print('Training Labels Shape:', train_labels.shape)
    # print('Testing Features Shape:', test_features.shape)
    # print('Testing Labels Shape:', test_labels.shape)

    return train_features, test_features, train_labels, test_labels

def baseline(test_features, feature_list, test_labels):
    # The baseline predictions are the historical averages
    baseline_preds = test_features[:, feature_list.index('average')]
    # Baseline errors, and display average baseline error
    baseline_errors = abs(baseline_preds - test_labels)
    # print('Average baseline error: ', round(np.mean(baseline_errors), 2))


def main():
    random.seed()
    features, labels, feature_list = loadData()
    train_features, test_features, train_labels, test_labels = splitData(features, labels)
    baseline(test_features, feature_list, test_labels)

    # Instantiate model with 1000 decision trees
    # rf = RandomForestRegressor(n_estimators=1000, random_state=42, max_features = "sqrt")
    m = 1000
    n = len(train_labels)#all training set
    d =  int(math.sqrt(len(feature_list)))
    print(d)
    myrf = RandomForest(5,d,m)
    myrf.fit(train_features, train_labels, feature_list)
    # Train the model on training data
    # rf.fit(train_features, train_labels);
    #
    # # Use the forest's predict method on the test data
    # predictions = rf.predict(test_features)
    # # Calculate the absolute errors
    # errors = abs(predictions - test_labels)
    # # Print out the mean absolute error (mae)
    # print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #
    # # Calculate mean absolute percentage error (MAPE)
    # mape = 100 * (errors / test_labels)
    # # Calculate and display accuracy
    # accuracy = 100 - np.mean(mape)
    # print('Accuracy:', round(accuracy, 2), '%.'
    print("a")

if __name__ == "__main__":
    main()