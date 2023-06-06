#####imports

import numpy as np
import os
import seaborn as sns
import scipy.stats as stat
from scipy.stats import pearsonr
from scipy.stats import pointbiserialr
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split


import acquire as acq

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from math import sqrt
import statsmodels.api as sm

from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor

### functions

def plot_hist_subplots(df):
    """
    Creates a subplot of histograms for each column in the dataframe.
    """
    # Set figure size.
    plt.figure(figsize=(16, 6))

    # Loop through columns.
    for i, col in enumerate(df.columns):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1

        # Create subplot.
        plt.subplot(2, 4, plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        plt.hist(df[col])

    # Display the plot.
    plt.show()
    
def prep_rrd_data(df):
    # Drop the specified columns
    df = df.drop(['railroad', 'carshzd', 'station', 'trkclas', 'typtrk', 'highspd', 'accdmg', 'totinj', 'totkld', 'county', 'stcnty', 'jointcd', 'region', 'year4', 'narr1', 'narr2', 'narr3', 'latitude', 'longitud'], axis=1)

    # Save the modified dataframe to a new csv file
    df.to_csv('RRD_US_combined_cleaned_reduced.csv', index=False)
    
    # Convert the 'year' column to a 4-digit year
    df['year'] = '20' + df['year'].astype(str)
    
    # Combine 'year', 'month', and 'day' columns into a single datetime column
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    # Move the 'date' column to be the second column
    cols = list(df.columns)
    cols.remove('date')
    cols.insert(1, 'date')
    df = df[cols]
    
    # Drop the 'year', 'month', and 'day' columns
    df = df.drop(['year', 'month', 'day'], axis=1)
    
    # Encode the 'cause' column
    df['cause'] = (df['cause'] == 'H').astype(int)
    
    return df

def split_data(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=None):
    """
    Splits a DataFrame into train, validation, and test sets.
    
    Parameters:
    - df: pandas DataFrame to be split
    - train_size: proportion of data to be used for training (default 0.7)
    - val_size: proportion of data to be used for validation (default 0.15)
    - test_size: proportion of data to be used for testing (default 0.15)
    - random_state: seed for the random number generator (default None)
    
    Returns:
    - train_df: pandas DataFrame containing the training set
    - val_df: pandas DataFrame containing the validation set
    - test_df: pandas DataFrame containing the test set
    """
    # Split the data into train and test sets
    train_df, test_df = train_test_split(df, train_size=train_size+val_size, test_size=test_size, random_state=random_state)
    
    # Split the train set into train and validation sets
    train_df, val_df = train_test_split(train_df, train_size=train_size/(train_size+val_size), test_size=val_size/(train_size+val_size), random_state=random_state)
    
    return train_df, val_df, test_df


