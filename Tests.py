import math

from RandomForest import RandomForest
from tree import loadFromBoth, ttsplit, accuracy, loadData


def emptyFile(label, fileName):
    f = open(f"{label}/{fileName}.csv", "w")
    f.write("")
    f.close()

def writeToFile(label, fileName, accurancy):
    f = open(f"{label}/{fileName}.csv", "a")
    f.write(str(accurancy))
    f.write("\n")
    f.close()

def averrageAccurancy(fileName, n, d, numberOfTrees, nomberOfForests, label, train, test_features, test_labels):
    averrageAcurrancy = 0
    emptyFile(label, fileName)
    for forest in range(nomberOfForests):
        rf = RandomForest(n, d, numberOfTrees)
        rf.fit(train, label)
        predictions = rf.predict(test_features)
        accurancy = accuracy(test_labels, predictions, label)

        averrageAcurrancy += accurancy
        writeToFile(label, fileName, round(accurancy, 2))
    return averrageAcurrancy / nomberOfForests


def main():
    workday_df, weekend_df = loadData("student-por.csv")
    # workday_df, weekend_df = loadFromBoth()
    label = "Walc"
    test_size = 0.4
    train, test = ttsplit(weekend_df, test_size)
    # train, test = ttsplit(workday_df, test_size)

    train_labels = train[[label]]
    train_features = train.drop([label], axis=1)
    test_labels = test[[label]]
    test_features = test.drop([label], axis=1)

    feature_list = list(train.columns)

    help = [20,40,60,80,100]
    for i in help:
        t = i
        n = len(train_labels)  # all training set
        # sqrt from features
        d = int(math.sqrt(len(feature_list) - 2))

        fileName = f"NumberOfTreesPor/out_n{n}_d{d}_t{t}"
        ave = averrageAccurancy(fileName, n, d, t, 10, label, train, test_features, test_labels)
        emptyFile(label, fileName+"Averrage")
        writeToFile(label, fileName+"Averrage", ave)


if __name__ == "__main__":
    main()