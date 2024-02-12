# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:10:16 2023

@author: sai
"""
import numpy as np
from numpy import array
from scipy.linalg import svd

# Define matrix A
A = array([[1, 0, 0, 0, 2],
           [0, 0, 3, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 4, 0, 0, 0]])
A

# Perform SVD on matrix A
U, d, Vt = svd(A)
U  # Left singular vectors
d  # Singular values
Vt  # Right singular vectors

# Convert singular values to a diagonal matrix
np.diag(d)


#applying svd to dataset
import pandas as pd

# Load dataset from an Excel file
data = pd.read_excel("c:/2-dataset/University_clustering.xlsx")

data.head()  # Display the first few rows of the dataset

# Remove non-numeric data assuming it's present in the first two columns
data = data.iloc[:, 2:]  # Keeping only numeric columns for analysis
data

# Import TruncatedSVD and apply it to the dataset
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=3)  # Reduce dimensions to 3 components
svd.fit(data)  # Fit TruncatedSVD on the dataset

# Transform the data and store the result in a DataFrame
result = pd.DataFrame(svd.transform(data))
result.head()  # Display the first few rows of the transformed data

# Rename columns of the resulting DataFrame
result.columns = ["pc0", "pc1", "pc2"]
result.head()  # Display the first few rows with renamed columns

# Scatter plot of pc0 vs pc1 from the resulting DataFrame
import matplotlib.pyplot as plt
plt.scatter(x=result.pc0, y=result.pc1)













