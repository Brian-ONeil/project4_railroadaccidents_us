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
    
def prep_rrd_data():
    '''pulls clean data, drops rows, converts datetime, maps codes to respective strings, encodes and makes dummy columns'''
    
    # pull clean data
    df = acq.clean_rrd()
    
    # Drop the specified columns
    df = df.drop(['railroad', 'carshzd', 'station', 'trkclas', 'typtrk', 'highspd', 'accdmg', 'totinj', 'totkld', 'county', 'stcnty', 'jointcd', 'region', 'year4', 'narr1', 'narr2', 'narr3', 'latitude', 'longitud'], axis=1)
    
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
    
   # Define a dictionary to map type codes to accident types
    type_map = {
        1: 'Derailment',
        2: 'Collision',
        3: 'Fire/explosion',
        4: 'Other',
        5: 'Unknown',
        6: 'Level crossing',
        7: 'Trespasser',
        8: 'Employee fatality',
        9: 'Grade crossing accident',
        10: 'Equipment derailment',
        11: 'Non-train accident',
        12: 'Hazardous material release'
    }

    # Replace the type codes with accident types
    df['type'] = df['type'].map(type_map)
    
    # Define a dictionary to map state codes to state names
    state_map = {
        1: 'Alabama',
        2: 'Alaska',
        3: 'Arizona',
        4: 'Arkansas',
        5: 'California',
        6: 'Colorado',
        7: 'Connecticut',
        8: 'Delaware',
        9: 'District of Columbia',
        10: 'Florida',
        11: 'Georgia',
        12: 'Hawaii',
        13: 'Idaho',
        14: 'Illinois',
        15: 'Indiana',
        16: 'Iowa',
        17: 'Kansas',
        18: 'Kentucky',
        19: 'Louisiana',
        20: 'Maine',
        21: 'Maryland',
        22: 'Massachusetts',
        23: 'Michigan',
        24: 'Minnesota',
        25: 'Mississippi',
        26: 'Missouri',
        27: 'Montana',
        28: 'Nebraska',
        29: 'Nevada',
        30: 'New Hampshire',
        31: 'New Jersey',
        32: 'New Mexico',
        33: 'New York',
        34: 'North Carolina',
        35: 'North Dakota',
        36: 'Ohio',
        37: 'Oklahoma',
        38: 'Oregon',
        39: 'Pennsylvania',
        40: 'Rhode Island',
        41: 'South Carolina',
        42: 'South Dakota',
        43: 'Tennessee',
        44: 'Texas',
        45: 'Utah',
        46: 'Vermont',
        47: 'Virginia',
        48: 'Washington',
        49: 'West Virginia',
        50: 'Wisconsin',
        51: 'Wyoming',
        52: 'Puerto Rico',
        53: 'Virgin Islands',
        54: 'Guam'
    }

    # Replace the state codes with state names
    df['state'] = df['state'].map(state_map)
    
    # Define a dictionary to map visibility codes to visibility conditions
    visibility_map = {
        1: '<=1/4 mile',
        2: '>1/4 mile and <=1/2 mile',
        3: '>1/2 mile and <=1 mile',
        4: '>1 mile and <=2 miles'
    }

    # Replace the visibility codes with visibility conditions
    df['visiblty'] = df['visiblty'].replace(visibility_map)
    
    # Define a dictionary to map weather codes to weather conditions
    weather_map = {
        1: 'Clear/PC',
        2: 'Rain',
        3: 'Snow/Hail',
        4: 'Fog/Smoke',
        5: 'Crosswinds',
        6: 'Blowing Dirt'
    }

    # Replace the weather codes with weather conditions
    df['weather'] = df['weather'].map(weather_map)
    
    # Define a dictionary with the mapping of numbers to maximum allowable speed limit strings
    acc_track_class = {
        1: 'Max spd 10mph or less',
        2: 'Max spd 10-20mph',
        3: 'Max spd 20-25mph',
        4: 'Max spd 25-40mph',
        5: 'Max spd 40-60mph',
        6: 'Max spd 60mph or more'
    }

    # Use the map() function to replace the numbers with their respective maximum allowable speed limit strings
    df['acctrkcl'] = df['acctrkcl'].map(acc_track_class)
    
    # Define a dictionary with the mapping of numbers to meanings
    acc_track = {
        1: 'Owned by Carrier',
        2: 'Leased Another Railroad',
        3: 'Jointly Owned',
        4: 'Trackage Rights Only'
    }
    # Use the map() function to replace the numbers with their respective meanings
    df['acctrk'] = df['acctrk'].map(acc_track)
    
    # Replace 'H' with 'Human' and everything else with 'Other'
    df['cause'] = df['cause'].apply(lambda x: 'Human' if x == 'H' else 'Other')
    
    # Encode and make dummy columns
    dummy_df = pd.get_dummies(df[['visiblty', 
                             'weather',
                             'acctrkcl',
                             'acctrk']],
                              drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)

    df = df.rename(columns=str.lower)
    
    # Save the modified dataframe to a new csv file
    df.to_csv('RRD_US_combined_cleaned_reduced.csv', index=False)
    
    return df

def split_data(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=123):
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


def get_X_train_val_test(train, validate, test, x_target, y_target):
    '''
    geting the X's and y's and returns them
    '''
    X_train = train.drop(columns = x_target)
    X_validate = validate.drop(columns = x_target)
    X_test = test.drop(columns = x_target)
    y_train = train[y_target]
    y_validate = validate[y_target]
    y_test = test[y_target]
    return X_train, X_validate, X_test, y_train, y_validate, y_test

def scaler_robust(X_train, X_validate, X_test):
    '''
    takes train, test, and validate data and uses the RobustScaler on it
    '''
    scaler = RobustScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_validate), scaler.transform(X_test)


def scaled_data_to_dataframe(X_train, X_validate, X_test):
    '''
    This function scales the data and returns it as a pandas dataframe
    '''
    X_train_columns = X_train.columns
    X_validate_columns = X_validate.columns
    X_test_columns = X_test.columns
    X_train_numbers, X_validade_numbers, X_test_numbers = scaler_robust(X_train, X_validate, X_test)
    X_train_scaled = pd.DataFrame(columns = X_train_columns)
    for i in range(int(X_train_numbers.shape[0])):
        X_train_scaled.loc[len(X_train_scaled.index)] = X_train_numbers[i]
    X_validate_scaled = pd.DataFrame(columns = X_validate_columns)
    for i in range(int(X_validade_numbers.shape[0])):
        X_validate_scaled.loc[len(X_validate_scaled.index)] = X_validade_numbers[i]
    X_test_scaled = pd.DataFrame(columns = X_test_columns)
    for i in range(int(X_test_numbers.shape[0])):
        X_test_scaled.loc[len(X_test_scaled.index)] = X_test_numbers[i]
    return X_train_scaled, X_validate_scaled, X_test_scaled

def X_train_data(train, val, test):
    # create X & y version of train/validate/test
    # where X contains the features we want to use and y is a series with just the target variable

    # Define the columns we want to use as features
    features = ['visiblty_>1 mile and <=2 miles', 
                'visiblty_>1/2 mile and <=1 mile', 
                'visiblty_>1/4 mile and <=1/2 mile', 
                'weather_clear/pc', 'weather_crosswinds', 
                'weather_fog/smoke', 'weather_rain', 
                'weather_snow/hail', 'trnspd', 'cdtrhr']

    # Create X and y datasets for the train set
    X_train = train[features]
    y_train = train['cause']

    # Create X and y datasets for the validate set
    X_val = val[features]
    y_val = val['cause']

    # Create X and y datasets for the test set
    X_test = test[features]
    y_test = test['cause']
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scaled_data(X_train, X_val, X_test, y_train, y_val, y_test):
   
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Fit the scaler to the training data
    scaler.fit(X_train)

    # Scale the training data
    X_train_scaled = scaler.transform(X_train)

    # Scale the validation data
    X_val_scaled = scaler.transform(X_val)

    # Scale the test data
    X_test_scaled = scaler.transform(X_test)

    # Return the scaled datasets
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test

