import math

from RandomForest import RandomForest
from tree import loadFromBoth, ttsplit


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

    numberOfTrees = 1000
    n = len(train_labels)  # all training set
    d = int(math.sqrt(len(feature_list)))
    print(len(feature_list))
    # myrf = RandomForest(5, d, numberOfTrees)
    #
    # rf = RandomForest(5, 3, 10)
    # rf.fit(train, label)
    # rf.predict(test_features)


if __name__ == "__main__":
    main()