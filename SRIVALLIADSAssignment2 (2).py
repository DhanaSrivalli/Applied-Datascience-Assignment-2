#!/usr/bin/env python
# coding: utf-8

# In[3]:


# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:01:59 2023

@author: DHANA SRIVALLI
"""
"""
This script generates visualizations of correlation matrices and bar plots for CO2 emissions data from the World Bank dataset.

Functions:

load_data: Load data from a CSV file into a pandas DataFrame.
return_df_dfT: Returns two data frames one with years as columns and the
                other one with countries as columns
pivoted_df: Generates the pivoted DataFrame to have the series names as columns
            and the country
Usage:

Load the World Bank dataset into a Pandas DataFrame
Call plot_heatmap, plot_bar_chart, or plot_line_chart with the DataFrame as input to generate visualizations
Note:
This script uses the following libraries: pandas, numpy, matplotlib
"""

# Importing the required libraries
# Loads the data from CSV file to pandas DataFrame




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def load_data(fname):
    '''
    Load data from a CSV file into a pandas DataFrame.

    Parameters
    ----------
    fname : String
        file name.

    Returns
    -------
   DataFrame
        DataFrame for the given file name

    '''
    return pd.read_csv(fname)
# Method for returning DF and Transposed DF


def return_df_dfT(fname):
    '''
    this function takes the file name as parameter and returns two data frames
    one with years as columns and the other one with countries as columns

    Parameters
    ----------
    fname : String
        DESCRIPTION.

    Returns
    -------
    df_years : DataFrame
        data frame years as columns.
    df_years_transposed : DataFrame
        data frame countries as columns.

    '''
    df = load_data(fname)

    df_selected = df.loc[(df['Country Name'].isin(countries))
                         & (df['Series Name'].isin(series))]

    # Create a DataFrame with years as columns
    df_years = df_selected.set_index(['Country Name', 'Series Name']).T

    # Replace all instances of ".." with NaN
    df_years.replace("..", np.nan, inplace=True)

    # Convert the values to numeric data type
    df_years = df_years.apply(pd.to_numeric, errors='coerce')

    # Describe the data
    # print("Data Description:")
    # print(df_years.describe())

    # Transpose the data for better visualization
    df_years_transposed = df_years.T
    return df_years, df_years_transposed

# Method for returning pivoted DataFrame


def pivoted_df(df, year):
    '''

    it takes a dataFrame and it selects the particular year data and generates
    the pivoted DataFrame to have the series names as columns and the country
    Parameters
    ----------
    df : DataFrame
        it takes the parameter of a data frame that should be pivoted.
    year : String
        Required year Column name.
    Returns
    -------
    df_pivoted : DataFrame
        returns the pivoted dataframe.

    '''
    df_filtered = df[df["Series Name"].isin(["Access to electricity (% of population)",
                                             "Access to electricity, rural (% of rural population)",
                                             "Access to electricity, urban (% of urban population)",
                                             "CO2 emissions (metric tons per capita)",
                                             "CO2 emissions (kg per PPP $ of GDP)",
                                             "CO2 emissions (kg per 2017 PPP $ of GDP)",
                                             "CO2 emissions (kg per 201 PPP $ of GDP)"])]

    # Pivot the DataFrame to have the series names as columns and the country
    # names as rows
    df_pivoted = df_filtered.pivot(
        index="Country Name", columns="Series Name", values=year)

    # Convert the values to numeric data type
    df_pivoted["CO2 emissions (metric tons per capita)"] = pd.to_numeric(
        df_pivoted["CO2 emissions (metric tons per capita)"])
    return df_pivoted


fileName = 'D:\\Datasets\\Metadata.csv'
countries = ['China', 'India', 'United States', 'Indonesia', 'Pakistan']
series = ['Access to electricity, rural (% of rural population)',
          'Access to electricity, urban (% of urban population)',
          'CO2 emissions (metric tons per capita)',
          'CO2 emissions (kg per PPP $ of GDP)',
          'CO2 emissions (kg per 2017 PPP $ of GDP)']
df_years, df_years_transposed = return_df_dfT(fileName)
print(df_years_transposed)


print("Data Description:")
print(df_years.describe())


# In[4]:


# Transpose the data for better visualization
df_years_transposed = df_years.T

# Calculate the statistical properties for each indicator
for indicator in df_years.columns:
    print(f"Indicator: {indicator}")
    print(f"Mean: {df_years[indicator].mean()}")
    print(f"Median: {df_years[indicator].median()}")
    print(f"Minimum: {df_years[indicator].min()}")
    print(f"Maximum: {df_years[indicator].max()}")
    print(f"Standard Deviation: {df_years[indicator].std()}")
    print()

# Calculate the correlation matrix for each country
corr_matrices = {}
for country in countries:
    corr_matrix = df_years[country].corr()
    corr_matrices[country] = corr_matrix

"""
This script generates two visualizations: a set of heatmaps displaying the correlations between
various development indicators for several countries, and a bar chart comparing the CO2 emissions of different countries.
The data is sourced from the World Bank Development Indicators.
"""

# Plot the correlation heatmap for Pakistan, India and Indonesia
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
countries = ['Pakistan', 'India', 'Indonesia']
colormaps = ['PuBuGn', 'BuPu', 'RdPu']
for i in range(3):
    heatmap = axes[i].pcolor(corr_matrices[countries[i]], cmap=colormaps[i])
    plt.colorbar(heatmap, ax=axes[i])
    axes[i].set_title(f'Correlation Heatmap - {countries[i]}', fontsize=14)
    axes[i].set_xlabel('Series Name', fontsize=12)
    axes[i].set_ylabel('Series Name', fontsize=12)

    if countries[i] == 'Pakistan':
        # Set yticks for Pakistan map
        axes[i].set_yticks(np.arange(len(series)))
        axes[i].set_yticklabels(series, fontsize=10)
    else:
        # Remove yticks for other maps
        axes[i].set_yticks([])

    # Set xticks for all maps
    axes[i].set_xticks(np.arange(len(series)))
    axes[i].set_xticklabels(series, rotation=90, fontsize=10)

    # Add values to heatmap
    for y in range(corr_matrices[countries[i]].shape[0]):
        for x in range(corr_matrices[countries[i]].shape[1]):
            axes[i].text(x + 0.5,
                         y + 0.5,
                         '{:.2f}'.format(corr_matrices[countries[i]].iloc[y,
                                                                          x]),
                         ha='center',
                         va='center',
                         fontsize=10)

plt.show()

df_pivoted = pivoted_df(load_data(fileName), '2019 [YR2019]')
# Set up colors and line style
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
line_styles = ['--', '-', ':', '-.', '--', '-', ':']

"""
Plots a bar chart comparing the CO2 emissions of different countries.

Args:
    df_pivoted (pandas.DataFrame): A pivoted DataFrame containing CO2 emissions data for different countries.
    colors (list): A list of colors to use for the bars.
    line_styles (list): A list of line styles to use for the dashed lines on top of each bar.

Returns:
    None
"""

# First bar chart
plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1, 2, 1)
bars1 = ax1.bar(
    x=df_pivoted.index,
    height=df_pivoted["CO2 emissions (metric tons per capita)"],
    color=colors)
plt.title("CO2 emissions (metric tons per capita)")
plt.xlabel("Country Name")
plt.ylabel("% of CO2 emissions (metric tons per capita)")
plt.xticks(rotation=90)

# Add dashed line on top of each bar
for i, bar in enumerate(bars1):
    ax1.axhline(y=bar.get_height(),
                color=colors[i], linestyle=line_styles[i], alpha=0.5)

# Second bar chart
ax2 = plt.subplot(1, 2, 2)
bars2 = ax2.bar(
    x=df_pivoted.index,
    height=df_pivoted["CO2 emissions (kg per 2017 PPP $ of GDP)"],
    color=colors)
plt.title("CO2 emissions (kg per 2017 PPP $ of GDP)")
plt.xlabel("Country Name")
plt.ylabel("CO2 emissions (kg per 2017 PPP $ of GDP)")
plt.xticks(rotation=90)

# Add dashed line on top of each bar
for i, bar in enumerate(bars2):
    ax2.axhline(y=bar.get_height(),
                color=colors[i], linestyle=line_styles[i], alpha=0.5)

plt.subplots_adjust(wspace=0.5)
plt.show()

plt.figure()
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df_pivoted["Access to electricity, rural (% of rural population)"],
        label="Access to electricity, rural")
ax.plot(df_pivoted["Access to electricity, urban (% of urban population)"],
        label="Access to electricity, urban")
ax.set_title("Access to electricity, rural vs urban", fontsize=14)
ax.set_xlabel("Country")
ax.set_ylabel("Access to electricity (%)")
ax.legend()

# Show the plot
plt.show()


# In[ ]:




