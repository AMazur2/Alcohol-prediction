import numpy as np
import pandas as pd
from pprint import pprint

class Tree:

    def __init__(self):
        pass

    # We check here whether our data set contains only data from one class
    def isPure(self, label) -> bool:
        uniques = np.unique(label)
        if len(uniques) == 1:
            return True
        return False

    # This function will be called only when data is pure, so we can return the classification of this data set
    def classify(self, label):
        l = label[0]
        return l[0]

    # Find potential splits
    def potentialSplits(self, data):
        splits = {}
        _, numberOfColumns = data.shape
        for column_id in range(numberOfColumns):
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
    def split(self, data, label, column, value):
        label_above = []
        label_below = []
        valuesInColumn = data[:, column]
        i = 0
        for featureValue in valuesInColumn:
            if featureValue <= value:
                label_below.append(label[i])
            else:
                label_above.append(label[i])
            i = i + 1

        return data[valuesInColumn <= value], label_below, data[valuesInColumn > value], label_above

    # Calculate entropy
    def entropy(self, data, label):
        _, counts = np.unique(label, return_counts=True)
        probabilities = counts / counts.sum()
        return sum(probabilities * -np.log2(probabilities))

    # calculate overall entropy for split
    def overallEntropy(self, data_below, label_below, data_above, label_above):
        numberOfPoints = len(data_below) + len(data_above)

        probabilityBelow = len(data_below) / numberOfPoints
        probabilityAbove = len(data_above) / numberOfPoints

        return probabilityBelow * self.entropy(data_below, label_below) + \
               probabilityAbove * self.entropy(data_above, label_above)

    # find best split
    def bestSplit(self, data, label, splits):
        splitEntropy = 100000
        splitColumn = 0
        splitValue = 0
        for column_id in splits:
            if bool(splits[column_id]):
                for value in splits[column_id]:
                    below, l_below, above, l_above = self.split(data, label, column_id, value)
                    currEntropy = self.overallEntropy(below, l_below, above, l_above)

                    if currEntropy < splitEntropy:
                        splitEntropy = currEntropy
                        splitValue = value
                        splitColumn = column_id

        return splitColumn, splitValue

    def checkDict(self, dict):
        empty = 0
        for key in dict.keys():
            if not bool(dict[key]):
                empty = empty + 1
        if empty == len(dict):
            return False
        else:
            return True

    def mostCommon(self, label_list):
        (uniques, counts) = np.unique(label_list, return_counts=True)

        mostCommonIndex = 0
        mostCommon = 0

        for i in range(len(uniques)):
            if mostCommon < counts[i]:
                mostCommonIndex = i
                mostCommon = counts[i]

        l = label_list[mostCommonIndex]
        return l[0]

    # Build decision tree where subtree is: {question: [yes_option, no_option]}
    def buildTree(self, df, labels, level):
        global columnHeaders
        if level == 0:  # we want to give whole dataframe to the algorithm so we must be sure that we have values
            data = df.values
            columnHeaders = df.columns
            label = labels.values
        else:
            data = df
            label = labels

        if self.isPure(label):
            return self.classify(label)
        else:
            level += 1
            splits = self.potentialSplits(data)
            if self.checkDict(splits):
                splitColumn, splitValue = self.bestSplit(data, label, splits)
                dataBelow, label_below, dataAbove, label_above = self.split(data, label, splitColumn, splitValue)

                question = "{} <= {}".format(columnHeaders[splitColumn], splitValue)
                subTree = {question: []}

                yes_option = self.buildTree(dataBelow, label_below, level)
                no_option = self.buildTree(dataAbove, label_above, level)

                if yes_option == no_option:
                    subTree[question].append(yes_option)
                else:
                    subTree[question].append(yes_option)
                    subTree[question].append(no_option)

                return subTree
            else:
                return self.mostCommon(label)

    def tryToClassify(self, individual, tree):
        if(isinstance(tree, float)):
            return tree
        question = list(tree.keys())[0]
        feature, operator, value = question.split()


        if individual[feature] <= float(value):
            answer = tree[question][0]
        else:
            try:
                answer = tree[question][1]
            except IndexError:
                answer = tree[question][0]

        if not isinstance(answer, dict):
            return answer
        else:
            subTree = answer
            return self.tryToClassify(individual, subTree)

    def testTree(self, tree, test_df, test_label):
        label_values = test_label.values
        values = label_values[:, 0]
        correct = 0
        for i in range(len(test_df)):
            label = values[i]
            classification = self.tryToClassify(test_df.iloc[i], tree)
            if classification == label:
                correct += 1

        return correct / (len(test_df))
