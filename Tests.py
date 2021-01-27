import math

from RandomForest import RandomForest
from tree import loadFromBoth, ttsplit, accuracy


def main():
    workday_df, weekend_df = loadFromBoth()
    label = "Dalc"
    test_size = 0.4
    # train, test = ttsplit(weekend_df, test_size)
    train, test = ttsplit(workday_df, test_size)

    train_labels = train[[label]]
    train_features = train.drop([label], axis=1)
    test_labels = test[[label]]
    test_features = test.drop([label], axis=1)

    feature_list = list(train.columns)

    numberOfTrees = 100
    n = len(train_labels)  # all training set
    #sqrt from features
    d = int(math.sqrt(len(feature_list)-2))

    # myrf = RandomForest(5, d, numberOfTrees)
    #
    rf = RandomForest(n, d, numberOfTrees)
    rf.fit(train, label)
    predictions = rf.predict(test_features)
    percent = accuracy(test_labels, predictions, label)
    print(percent)


if __name__ == "__main__":
    main()