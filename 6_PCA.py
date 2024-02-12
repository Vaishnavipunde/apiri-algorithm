# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:42:53 2023

@author: sai
"""

# Import necessary libraries
import pandas as pd  # Library for data manipulation and analysis
import numpy as np   # Library for numerical operations
import matplotlib.pyplot as plt  # Library for data visualization

# Read the Excel file "University_clustering.xlsx" into a DataFrame named uni1
uni1 = pd.read_excel("c:/2-dataset/University_clustering.xlsx")

# Display descriptive statistics of numerical columns in the DataFrame uni1
uni1.describe()

# Display concise summary information about the DataFrame uni1
uni1.info()

# Remove the column named "State" from the DataFrame uni1 and store the modified DataFrame in uni
uni = uni1.drop(["State"], axis=1)

# Import necessary libraries
from sklearn.decomposition import PCA  # Library for performing Principal Component Analysis
import matplotlib.pyplot as plt  # Library for data visualization
from sklearn.preprocessing import scale  # Library for data scaling

# Consider only numerical data by excluding the first column (assuming it's non-numeric or an identifier)
uni.data = uni.iloc[:, 1:]

# Normalize the numerical data using the scale function from sklearn.preprocessing
uni_norma1 = scale(uni.data)


# Initialize PCA with the number of components set to 6
pca = PCA(n_components=6)

# Fit the PCA model and transform the normalized data into principal components
pca_values = pca.fit_transform(uni_norma1)

# Get the explained variance ratio of each principal component
var = pca.explained_variance_ratio_
var

# Calculate cumulative explained variance
var1 = np.cumsum(np.round(var, decimals=4) * 100)
var1

# Plot the cumulative explained variance
plt.plot(var1, color="red")

# pca_values contains the transformed data after applying PCA
pca_values





