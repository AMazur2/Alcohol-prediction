import numpy as np
import pandas as pd
from pprint import pprint
from random import sample

# Prepare data - all data in int
from Tree import Tree


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


def main():
    # Let's load a data
    # workday_df, weekend_df = loadData("student-mat.csv")
    # workday_df, weekend_df = loadData("student-por.csv")
    workday_df, weekend_df = loadFromBoth()

    # "Dalc" or "Walc" depends on df
    label = "Dalc"

    # Now we split data into train_df and test_df - we can split either workday_df or weekend_df
    # second parameter is test data size - between 0 and 1
    train, test = ttsplit(workday_df, 0.4)

    train_labels = train[[label]]
    train_df = train.drop([label], axis=1)
    test_labels = test[[label]]
    test_df = test.drop([label], axis=1)

    # main algorithm
    tree = Tree()
    newTree = tree.buildTree(train_df, train_labels, 0)
    pprint(newTree)

    result = tree.testTree(newTree, test_df, test_labels)
    print(result)


if __name__ == "__main__":
    main()
