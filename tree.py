import numpy as np
import pandas as pd
from pprint import pprint
from random import sample

# Prepare data - all data in int
from Tree import Tree
from RandomForest import RandomForest

def prepareData(df):
    dataType = df.dtypes.tolist()  # we need to know types of values in every column
    columnNames = df.columns.tolist()  # save the column names
    values = df.to_numpy()

    for i in range(len(dataType)):
        if dataType[i] == 'object':
            inColumn = values[:, i]
            unique = np.unique(inColumn)

            for j in range(len(unique)):
                for k in range(len(inColumn)):
                    if inColumn[k] == unique[j]:
                        inColumn[k] = j

    df = pd.DataFrame(values, columns=columnNames).astype('float')
    return df


# Load and prepare data from specific file
def loadData(filename):
    df = pd.read_csv(filename)
    df = prepareData(df)
    workday = df.drop(['Walc'], axis=1)
    weekend = df.drop(['Dalc'], axis=1)

    return workday, weekend


# Load and prepare data from both files
def loadFromBoth():
    df1 = pd.read_csv("student-mat.csv")
    df1["course"] = "mat"
    df2 = pd.read_csv("student-por.csv")
    df2["course"] = "por"
    df = pd.concat([df1, df2], ignore_index=True)
    df = prepareData(df)
    workday = df.drop(['Walc'], axis=1)
    weekend = df.drop(['Dalc'], axis=1)
    return workday, weekend


# Train-Test Split
def ttsplit(df, test_size):
    size = round(test_size * len(df))
    index = df.index.tolist()
    test_indexes = sample(population=index, k=size)

    test_df = df.loc[test_indexes]
    train_df = df.drop(test_indexes)
    return train_df, test_df


def splitInputOutput(data: pd.DataFrame):
    numberOfColumns: int = len(data.columns)
    numberOfRows: int = len(data)
    x = data.drop(['Dalc'], axis = 1)
    y = data.Dalc
    return x,y

def accuracy(test_labels, predictions, label):
    counter=0
    for i in range(len(test_labels)):
        # print(i)
        # print(test_labels.iloc[i])
        # print(test_labels.iloc[i]["Dalc"], "\n")
        dalc = test_labels.iloc[i]["Dalc"]
        if(dalc == predictions[i]):
            counter += 1
    return 100 * counter/len(test_labels)

def main():
    # Let's load a data
    # workday_df, weekend_df = loadData("student-mat.csv")
    # workday_df, weekend_df = loadData("student-por.csv")
    workday_df, weekend_df = loadFromBoth()

    # "Dalc" or "Walc" depends on df
    # label = "Walc"
    label = "Dalc"

    # Now we split data into train_df and test_df - we can split either workday_df or weekend_df
    # second parameter is test data size - between 0 and 1

    test_size = 0.4
    # train, test = ttsplit(weekend_df, test_size)
    train, test = ttsplit(workday_df, test_size)

    train_labels = train[[label]]
    train_features = train.drop([label], axis=1)
    test_labels = test[[label]]
    test_features = test.drop([label], axis=1)



    tree = Tree()
    newTree = tree.buildTree(train_features, train_labels, 0)

    rf = RandomForest(5, 3, 10)
    rf.fit(train, label)
    predictions = rf.predict(test_features)

    percent = accuracy(test_labels, predictions, label)
    print(percent)


if __name__ == "__main__":
    main()
