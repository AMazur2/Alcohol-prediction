import numpy as np
import pandas as pd
from pprint import pprint
from random import sample


# Prepare data - all data in int
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
def loadData(filename):  # TODO: zrobic podzial lepszym sposobem niz wypisaniem wszystkiego
    df = pd.read_csv(filename)
    df = prepareData(df)
    workday = df[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
                  'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities',
                  'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health', 'absences',
                  'G1', 'G2', 'G3', 'Dalc']]
    weekend = df[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
                  'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities',
                  'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health', 'absences',
                  'G1', 'G2', 'G3', 'Walc']]

    return workday, weekend


# Load and prepare data from both files
def loadFromBoth():
    df1 = pd.read_csv("student-mat.csv")
    df1["course"] = "mat"
    df2 = pd.read_csv("student-por.csv")
    df2["course"] = "por"
    df = pd.concat([df1, df2], ignore_index=True)
    df = prepareData(df)
    workday = df[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
                  'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities',
                  'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health', 'absences',
                  'G1', 'G2', 'G3', 'course', 'Dalc']]
    weekend = df[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
                  'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities',
                  'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health', 'absences',
                  'G1', 'G2', 'G3', 'course', 'Walc']]
    return workday, weekend


# Train-Test Split
def ttsplit(df, test_size):
    size = round(test_size * len(df))
    index = df.index.tolist()
    test_indexes = sample(population=index, k=size)

    test_df = df.loc[test_indexes]
    train_df = df.drop(test_indexes)
    return train_df, test_df


# We check here whether our data set contains only data from one class
def isPure(data) -> bool:
    label = data[:, -1]
    uniques = np.unique(label)
    if len(uniques) == 1:
        return True
    return False


# This function will be called only when data is pure, so we can return the classification of this data set
def classify(data):
    label = data[:, -1]
    classification = label[0]
    return classification


# Find potential splits
def potentialSplits(data):
    splits = {}
    _, numberOfColumns = data.shape
    for column_id in range(numberOfColumns - 1):
        splits[column_id] = []
        values = data[:, column_id]
        unique = np.sort(np.unique(values))

        for index in range(len(unique)):
            if index != 0:
                currentVal = unique[index]
                previousVal = unique[index - 1]
                splits[column_id].append(float((currentVal + previousVal) / 2))

    return splits


# Split data by column and value in it
def split(data, column, value):
    valuesInColumn = data[:, column]
    return data[valuesInColumn <= value], data[valuesInColumn > value]


# Calculate entropy
def entropy(data):
    label = data[:, -1]
    _, counts = np.unique(label, return_counts=True)
    probabilities = counts / counts.sum()
    return sum(probabilities * -np.log2(probabilities))


# calculate overall entropy for split
def overallEntropy(data_below, data_above):
    numberOfPoints = len(data_below) + len(data_above)

    probabilityBelow = len(data_below) / numberOfPoints
    probabilityAbove = len(data_above) / numberOfPoints

    return probabilityBelow * entropy(data_below) + probabilityAbove * entropy(data_above)


# find best split
def bestSplit(data, splits):
    splitEntropy = 100000
    splitColumn = 0
    splitValue = 0
    for column_id in splits:
        for value in splits[column_id]:
            below, above = split(data, column_id, value)
            currEntropy = overallEntropy(below, above)

            if currEntropy < splitEntropy:
                splitEntropy = currEntropy
                splitValue = value
                splitColumn = column_id

    return splitColumn, splitValue


# Build decision tree where subtree is: {question: [yes_option, no_option]}
def buildTree(df, level):
    global columnHeaders
    if level == 0:  # we want to give whole dataframe to the algorithm so we must be sure that we have values
        data = df.values
        columnHeaders = df.columns
    else:
        data = df

    if isPure(data):
        return classify(data)
    else:
        level += 1
        splits = potentialSplits(data)
        splitColumn, splitValue = bestSplit(data, splits)
        dataBelow, dataAbove = split(data, splitColumn, splitValue)

        question = "{} <= {}".format(columnHeaders[splitColumn], splitValue)
        subTree = {question: []}

        yes_option = buildTree(dataBelow, level)
        no_option = buildTree(dataAbove, level)

        if yes_option == no_option:
            subTree[question].append(yes_option)
        else:
            subTree[question].append(yes_option)
            subTree[question].append(no_option)

        return subTree


def tryToClassify(individual, tree):
    question = list(tree.keys())[0]
    feature, operator, value = question.split()

    if individual[feature] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    if not isinstance(answer, dict):
        return answer
    else:
        subTree = answer
        return tryToClassify(individual, subTree)


def testTree(tree, test_df):

    correct = 0
    for i in range(len(test_df)):
        label = test_df.iloc[i][-1]
        classification = tryToClassify(test_df.iloc[i], tree)
        if classification == label:
            correct += 1

    return correct/(len(test_df))



def main():
    # Let's load a data
    workday_df, weekend_df = loadData("student-mat.csv")
    # workday_df, weekend_df = loadData("student-por.csv")
    # workday_df, weekend_df = loadFromBoth()

    # Now we split data into train_df and test_df - we can split either workday_df or weekend_df
    # second parameter is test data size - between 0 and 1
    train_df, test_df = ttsplit(workday_df, 0.4)

    # main algorithm
    tree = buildTree(train_df, 0)
    pprint(tree)

    result = testTree(tree, test_df)


if __name__ == "__main__":
    main()
