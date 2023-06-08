### imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import datetime
from datetime import timedelta, datetime

# .py files
import acquire as acq
import prepare as prep
import explore as exp

#stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score
from scipy.stats import mannwhitneyu

### functions

def visibility_plot(data):
    plt.figure(figsize=(12,8))
    sns.countplot(x='visiblty', hue='cause', data=data, palette=['green', 'gold'])
    plt.title('Does visibility affect human factor accidents?')
    plt.xlabel('Visibility')
    plt.ylabel('Human Factor Caused Accidents')
    plt.show()
    
def visibility_stat(data):
    # Create a contingency table of cause and visibility
    contingency_table = pd.crosstab(data['cause'], data['visiblty'])

    # Perform a chi-squared test of independence
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Print the results of the chi-squared test
    print('Chi-squared statistic:', chi2)
    print('p-value:', p)
    
def weather_plot(data):
    plt.figure(figsize=(12,8))
    sns.countplot(x='weather', hue='cause', data=data, palette=['green', 'gold'])
    plt.title('Does weather affect human factor accidents?')
    plt.xlabel('Weather')
    plt.ylabel('Human Factor Caused Accidents')
    plt.show()
    
def weather_stat(data):
    # Create a contingency table of cause and weather
    contingency_table = pd.crosstab(data['cause'], data['weather'])

    # Perform a chi-squared test of independence
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Print the results of the chi-squared test
    print('Chi-squared statistic:', chi2)
    print('p-value:', p)
    
def cdthr_plot(data):
    sns.violinplot(x='cause', y='cdtrhr', data=data, palette=['green', 'gold'])
    plt.title('Does conductor work hours affect human factor caused accidents?')
    plt.xlabel('Cause')
    plt.ylabel('Conductor Hours')
    plt.show()
    
def cdtrhr_stat(data):
    # Split the data into two groups based on the 'cause' column
    human_hours = data.loc[data['cause'] == 'Human', 'cdtrhr']
    other_hours = data.loc[data['cause'] != 'Human', 'cdtrhr']

    # Perform the Mann-Whitney U test
    stat, p = mannwhitneyu(human_hours, other_hours)

    # Print the results
    print('Mann-Whitney U test results:')
    print(f'Statistic: {stat}')
    print(f'p-value: {p}')
    
def trnspd_plot(data):
    sns.violinplot(x='cause', y='trnspd', data=data, palette=['green', 'gold'])
    plt.title('Does train speed affect human factor caused accidents?')
    plt.xlabel('Cause')
    plt.ylabel('Train Speed')
    plt.show()
    
def trnspd_stat(data):
    # Split the data into two groups based on the 'cause' column
    human_speeds = data.loc[data['cause'] == 'Human', 'trnspd']
    other_speeds = data.loc[data['cause'] != 'Human', 'trnspd']

    # Perform the Mann-Whitney U test
    stat, p = mannwhitneyu(human_speeds, other_speeds)

    # Print the results
    print('Mann-Whitney U test results:')
    print(f'Statistic: {stat}')
    print(f'p-value: {p}')
    
def DTC_model(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test):
    '''send in scaled train then run DTC model'''
    dt = DecisionTreeClassifier()

    # Define the hyperparameters to search
    params = {'max_depth': [2, 4, 6, 8, 10],
              'min_samples_split': [2, 4, 6, 8, 10],
              'min_samples_leaf': [1, 2, 3, 4, 5]}

    # Create a grid search object
    grid_search = GridSearchCV(dt, params, cv=5)

    # Fit the grid search object on the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Create a decision tree model with the best hyperparameters
    dt = DecisionTreeClassifier(max_depth=best_params['max_depth'],
                                min_samples_split=best_params['min_samples_split'],
                                min_samples_leaf=best_params['min_samples_leaf'])

    # Fit the model on the training data
    dt.fit(X_train_scaled, y_train)

    # Predict the target variable for the training and validation data
    y_train_pred = dt.predict(X_train_scaled)
    y_val_pred = dt.predict(X_val_scaled)

    # Calculate the accuracy of the model on the training and validation data
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    # Print the accuracy of the model on the training and validation data
    print('Training Accuracy:', train_accuracy)
    print('Validation Accuracy:', val_accuracy)
    
    

def RF_model(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test):
    '''send in scaled train then run RF model'''

    # Create a random forest model
    rf = RandomForestClassifier()
    ​
    # Define the hyperparameters to search
    params = {'n_estimators': [50, 100, 150, 200],
              'max_depth': [2, 4, 6, 8, 10],
              'min_samples_split': [2, 4, 6, 8, 10],
              'min_samples_leaf': [1, 2, 3, 4, 5]}
    ​
    # Create a grid search object
    grid_search = GridSearchCV(rf, params, cv=5)
    ​
    # Fit the grid search object on the training data
    grid_search.fit(X_train_scaled, y_train)
    ​
    # Get the best hyperparameters
    best_params = grid_search.best_params_
    ​
    # Create a random forest model with the best hyperparameters
    rf = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                max_depth=best_params['max_depth'],
                                min_samples_split=best_params['min_samples_split'],
                                min_samples_leaf=best_params['min_samples_leaf'])
    ​
    # Fit the model on the training data
    rf.fit(X_train_scaled, y_train)
    ​
    # Predict the target variable for the training and validation data
    y_train_pred = rf.predict(X_train_scaled)
    y_val_pred = rf.predict(X_val_scaled)
    ​
    # Calculate the accuracy of the model on the training and validation data
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    ​
    # Print the accuracy of the model on the training and validation data
    print('Training Accuracy:', train_accuracy)
    print('Validation Accuracy:', val_accuracy)
    
    
