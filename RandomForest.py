import numpy as np
import pandas as pd





class RandomForest:

    #Number of samples in one tree
    #Number of features in one tree
    #Number of trees

    def __init__(self, n: int, d: int, m: int):
        self.n = n
        self.d = d
        self.m = m
        self.trees = []

    # x - features on wich I predict y - feature I want to predict
    def fit(self, xTrain, yTrain):
        # for i in range(self.m):
        chosenSamples = self.randomSamples(self.n, xTrain)
        readyDataset = np.array([1])
        # print(readyDataset)
        # np.delete(readyDataset,0)
        # print(readyDataset)
        for chosenSample in chosenSamples:
            np.append(readyDataset,[self.randomFeatures(self.d, chosenSample)])

        print(readyDataset)
            # cd = randomFeatures(d,)
        train_df = pd.DataFrame(data=readyDataset[0:, 0:])  # values
                    # index = data[1:, 0],  # 1st column as index
                    # columns = data[0, 1:])
        print(train_df)



    def predict(self, x, y):
        pass

    def randomSamples(self, n, xTrain):
        number_of_rows = xTrain.shape[0]
        random_indices = np.random.choice(number_of_rows, size=n, replace=False)
        random_rows = xTrain[random_indices, :]
        # random_rows = np.array(random_rows)
        # print(type(random_rows))
        return random_rows

    def randomFeatures(self, d, chosenSample):
        number_of_rows = chosenSample.shape[0]
        random_indices = np.random.choice(number_of_rows, size=d, replace=False)
        random_rows = chosenSample[random_indices]
        # print(random_rows)
        return random_rows