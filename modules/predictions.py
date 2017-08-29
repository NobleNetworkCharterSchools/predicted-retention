#!python3
"""
Class for doing regression analyses and creating persistence predictions
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

class Prediction():
    """Class to work with raw data and create regressions and other functions"""

    def to_csv(self, file_name):
        """Saves the main df to a csv"""
        self.df.to_csv(file_name)

    def describe(self):
        """Prints some summary stats"""
        print(self.title)
        print('{} training items, {} testing items'.format(
            len(self.train_df), len(self.test_df)))
        print(self.df.info())
        print(self.description)

    def make_coef_df(self, trial_name):
        """Creates a single row DataFrame coefficients"""
        out_df = pd.DataFrame(self.coefs, columns=['Field', trial_name])
        out_df = out_df.set_index(['Field'])
        return out_df.transpose()

    def __init__(self, raw_df, years, X, y, title,
            require=None, remove=None, train='RandomSplit'):
        """
        df=full raw dataframe from input files
        years=list of Classes to include in analysis
        X=list of column labels as independent variables
        Y=column label of dependent variable (must be binary)
        require=None or a list of tuples with ('Column',1 or 0) to thin data
        remove=None or list of tuples with ('Column',[x0, ..., xn]) to remove
        title=text description of this case
        train=training method default of 'RandomSplit' will use that
              column from the data and train on 1, test on 0; other options
              are None which uses all data for training and testing and
              a list of class years, which specifies the training set with
              the testing set being all other years
        """
        self.title = title
        self.X = X
        self.y = y
        all_columns = X.copy()
        all_columns.append(y)
        all_columns.append('Class')
        if train=='RandomSplit':
            all_columns.append('RandomSplit')

        # Get core df from raw data
        self.df = raw_df.loc[raw_df['Class'].isin(years)]
        if require:
            for c, val in require:
                self.df = self.df.loc[self.df[c] == val]
        if remove:
            for c, vals in remove:
                self.df = self.df.loc[~self.df[c].isin(vals)]

        self.df = self.df.filter(all_columns).dropna()

        # Split training and testing sets
        if not train:
            self.train_df = self.df.copy()
            self.test_df = self.df.copy()
        elif train == 'RandomSplit':
            self.train_df = self.df.loc[self.df['RandomSplit'] == 1]
            self.test_df = self.df.loc[self.df['RandomSplit'] == 0]
        else:
            self.train_df = self.df.loc[self.df['Class'].isin(train)]
            self.test_df = self.df.loc[~self.df['Class'].isin(train)]

        # create statistics
        self.logreg = LogisticRegression(C=1e9)
        self.logreg.fit(self.train_df[X], self.train_df[y].values.ravel())
        self.coefs = list(zip(X, self.logreg.coef_[0]))
        self.coefs.append(('intercept', self.logreg.intercept_.tolist()[0]))
        test_positive = len(self.test_df[self.test_df[y] == 1])
        test_n = len(self.test_df)
        train_positive = len(self.train_df[self.train_df[y] == 1])
        train_n = len(self.train_df)
        if test_n == test_positive: # No AOC if 100% positive
            self.test_aoc_score = -1
        else:
            self.test_aoc_score = roc_auc_score(self.test_df[y],
                self.logreg.predict(self.test_df[X]))
        if train_n == train_positive:
            self.train_aoc_score = -1
        else:
            self.train_aoc_score = roc_auc_score(self.train_df[y],
                self.logreg.predict(self.train_df[X]))
                            
        train_ci = np.sqrt(train_positive/(train_n-train_positive)/train_n)
        self.description = {
                'Train n': train_n,
                'Train %positive': train_positive/train_n,
                'Train score': self.logreg.score(self.train_df[X],
                                                 self.train_df[y]),
                'Train AOC': self.train_aoc_score,
                'Train CI': train_ci,
                'Test n': test_n,
                'Test %positive': test_positive/test_n,
                'Test score': self.logreg.score(self.test_df[X],
                                                 self.test_df[y]),
                'Test AOC': self.test_aoc_score,
                'Coefs': str(self.coefs),
                }
        self.desc_df = pd.DataFrame(self.description,index=[self.title])
