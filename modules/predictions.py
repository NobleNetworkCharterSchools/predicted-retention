#!python3
"""
Class for doing regression analyses and creating persistence predictions
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class Prediction():
    """Class to work with raw data and create regressions and other functions"""

    def to_csv(self, file_name):
        """Saves the main df to a csv"""
        self.df.to_csv(file_name)

    def describe(self):
        """Prints some summary stats"""
        print(self.title)
        print(self.df.info())

    def __init__(self, raw_df, years, X, Y, title,
            require=None, train='RandomSplit'):
        """
        df=full raw dataframe from input files
        years=list of Classes to include in analysis
        X=list of column labels as independent variables
        Y=column label of dependent variable (must be binary)
        require=None or a list of tuples with ('Column',1 or 0) to thin data
        title=text description of this case
        train=training method default of 'RandomSplit' will use that
              column from the data and train on 1, test on 0; other options
              are None which uses all data for training and testing and
              a list of class years, which specifies the training set with
              the testing set being all other years
        """
        self.title = title
        self.X = X
        self.Y = Y
        all_columns = X.copy()
        all_columns.append(Y)
        all_columns.append('Class')
        if train=='RandomSplit':
            all_columns.append('RandomSplit')
        self.df = raw_df.loc[raw_df['Class'].isin(years)]
        if require:
            for c, val in require:
                self.df = self.df.loc[self.df[c] == val]
        self.df = self.df.filter(all_columns).dropna()

