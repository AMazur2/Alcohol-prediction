import numpy as np
import pandas as pd

class Tree:

    def __init__(self, labelColumn: int):
        self.labelColumn = labelColumn

    # We check here whether our data set contains only data from one class
    def isPure(self, data) -> bool:
        label = data[:, self.labelColumn]
        uniques = np.unique(label)
        if len(uniques) == 1:
            return True
        return False

    # This function will be called only when data is pure, so we can return the classification of this data set
    def classify(self, data):
        label = data[:, self.labelColumn]
        classification = label[0]
        return classification

    # Find potential splits
    def potentialSplits(self, data):
        splits = {}
        _, numberOfColumns = data.shape
        for column_id in range(numberOfColumns):
            if column_id != self.labelColumn:
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
    def split(self, data, column, value):
        valuesInColumn = data[:, column]
        return data[valuesInColumn <= value], data[valuesInColumn > value]

    # Calculate entropy
    def entropy(self, data):
        label = data[:, self.labelColumn]
        _, counts = np.unique(label, return_counts=True)
        probabilities = counts / counts.sum()
        return sum(probabilities * -np.log2(probabilities))

    # calculate overall entropy for split
    def overallEntropy(self, data_below, data_above):
        numberOfPoints = len(data_below) + len(data_above)

        probabilityBelow = len(data_below) / numberOfPoints
        probabilityAbove = len(data_above) / numberOfPoints

        return probabilityBelow * self.entropy(data_below) + probabilityAbove * self.entropy(data_above)

    # find best split
    def bestSplit(self, data, splits):
        splitEntropy = 100000
        splitColumn = 0
        splitValue = 0
        for column_id in splits:
            for value in splits[column_id]:
                below, above = self.split(data, column_id, value)
                currEntropy = self.overallEntropy(below, above)

                if currEntropy < splitEntropy:
                    splitEntropy = currEntropy
                    splitValue = value
                    splitColumn = column_id

        return splitColumn, splitValue

    # Build decision tree where subtree is: {question: [yes_option, no_option]}
    def buildTree(self, df, level):
        global columnHeaders
        if level == 0:  # we want to give whole dataframe to the algorithm so we must be sure that we have values
            data = df.values
            columnHeaders = df.columns
        else:
            data = df

        if self.isPure(data):
            return self.classify(data)
        else:
            level += 1
            splits = self.potentialSplits(data)
            splitColumn, splitValue = self.bestSplit(data, splits)
            dataBelow, dataAbove = self.split(data, splitColumn, splitValue)

            question = "{} <= {}".format(columnHeaders[splitColumn], splitValue)
            subTree = {question: []}

            yes_option = self.buildTree(dataBelow, level)
            no_option = self.buildTree(dataAbove, level)

            if yes_option == no_option:
                subTree[question].append(yes_option)
            else:
                subTree[question].append(yes_option)
                subTree[question].append(no_option)

            return subTree

    def tryToClassify(self, individual, tree):
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
            return self.tryToClassify(individual, subTree)

    def testTree(self, tree, test_df):

        correct = 0
        for i in range(len(test_df)):
            label = test_df.iloc[i][self.labelColumn]
            classification = self.tryToClassify(test_df.iloc[i], tree)
            if classification == label:
                correct += 1

        return correct / (len(test_df))
