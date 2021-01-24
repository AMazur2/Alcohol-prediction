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

    # x - features on wich I predict y - feature I want to predict
    def fit(self, xTrain, yTrain):
        pass


    def predict(self, x, y):
        pass