### imports

import pandas as pd
import numpy as np
import matplotlib

### functions

def clean_rrd():
    # Read in the first CSV file and keep only the column titles
    df = pd.read_csv('RRD_US_2023.csv', nrows=0)

    # Loop through the remaining CSV files and stack them on top of the first one
    for year in range(2022, 2018, -1):
        filename = f'RRD_US_{year}.csv'
        temp_df = pd.read_csv(filename, header=0)
        df = pd.concat([df, temp_df], axis=0, ignore_index=True)

    # Write the combined CSV file to disk
    df.to_csv('RRD_US_combined.csv', index=False)
    
    # Drop the columns with more than 50% nulls
    # Calculate the percentage of null values in each column
    null_percentages = df.isnull().sum() / len(df) * 100

    # Get the column names where the null percentage is greater than 50%
    columns_to_drop = null_percentages[null_percentages > 50].index

    # Drop the columns from the dataframe
    df.drop(columns_to_drop, axis=1, inplace=True)

    # Write the updated dataframe to a new CSV file
    df.to_csv('RRD_US_combined_cleaned.csv', index=False)
    
    # Convert the column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Select the desired columns
    columns_to_keep = ['railroad', 'incdtno', 'year', 'month', 'day', 'timehr', 'timemin', 'ampm', 'type', 'carshzd', 'station', 'state', 'temp', 'visiblty', 'weather', 'trnspd', 'typspd', 'tons', 'trkclas', 'typtrk', 'loadf1', 'emptyf1', 'cause', 'acctrk', 'acctrkcl', 'highspd', 'accdmg', 'stcnty', 'totinj', 'totkld', 'enghr', 'cdtrhr', 'jointcd', 'region', 'year4', 'county', 'cntycd', 'narr1', 'narr2', 'narr3', 'latitude', 'longitud']
    df = df[columns_to_keep]

    # Write the updated dataframe to a new CSV file
    df.to_csv('RRD_US_combined_cleaned.csv', index=False)
    
    # Delete rows containing null values for specified columns
    df = df.dropna(subset=['state', 'visiblty', 'weather', 'trkclas', 'typtrk', 'cause', 'acctrk', 'acctrkcl', 'stcnty', 'region', 'county', 'cntycd'])
    
    # Drop the 'typspd' column
    df = df.drop('typspd', axis=1)
    
    # Convert columns to integers
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)
    df['timehr'] = df['timehr'].astype(int)
    df['timemin'] = df['timemin'].astype(int)
    df['carshzd'] = df['carshzd'].astype(int)
    df['temp'] = df['temp'].astype(int)
    df['visiblty'] = df['visiblty'].astype(int)
    df['trnspd'] = df['trnspd'].astype(int)
    df['tons'] = df['tons'].astype(int)
    
    # Replace 'X' with '1' in the trkclas and acctrkcl columns
    df['trkclas'] = df['trkclas'].str.replace('X', '1')
    df['acctrkcl'] = df['acctrkcl'].str.replace('X', '1')
    
    # Delete rows containing 'O' in the trkclas and acctrkcl columns
    df = df[~df['trkclas'].str.contains('O')]
    df = df[~df['acctrkcl'].str.contains('O')]
    
    # Convert more columns to integers
    df['trkclas'] = df['trkclas'].astype(int)
    df['loadf1'] = df['loadf1'].astype(int)
    df['emptyf1'] = df['emptyf1'].astype(int)
    df['acctrk'] = df['acctrk'].astype(int)
    df['acctrkcl'] = df['acctrkcl'].astype(int)
    df['highspd'] = df['highspd'].astype(int)
    df['accdmg'] = df['accdmg'].astype(int)
    df['year4'] = df['year4'].astype(int)
    df['cntycd'] = df['cntycd'].astype(int)
    
    # Replace 'nan' values in the 'enghr' and 'cdtrhr' columns with the mode from those columns
    enghr_mode = df['enghr'].mode()[0]
    cdtrhr_mode = df['cdtrhr'].mode()[0]
    df['enghr'] = df['enghr'].fillna(enghr_mode)
    df['cdtrhr'] = df['cdtrhr'].fillna(cdtrhr_mode)
    
    # Convert more columns to integers
    df['enghr'] = df['enghr'].astype(int)
    df['cdtrhr'] = df['cdtrhr'].astype(int)
    df['type'] = df['type'].astype(int)
    df['state'] = df['state'].astype(int)
    df['totinj'] = df['totinj'].astype(int)
    df['totkld'] = df['totkld'].astype(int)
    df['jointcd'] = df['jointcd'].astype(int)
    df['region'] = df['region'].astype(int)
    
    # Extract the first letter from the 'cause' column
    df['cause'] = df['cause'].str[0]
    
    return df
