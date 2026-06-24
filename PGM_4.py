# Data Exploration

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load dataset
df = pd.read_csv("/content/Healthcare-Diabetes.csv")
#https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes

# Basic information
print(df.head())
print("\nRows and Columns:", df.shape)
print("\nColumns:", df.columns.tolist())

# Dataset details
print(df.info())
print(df.describe())

# Missing and duplicate values
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# Histograms
df.hist(figsize=(12,12))
plt.show()

# Outcome distribution
if 'Outcome' in df.columns:
    sns.countplot(x='Outcome', data=df)
    plt.title("Outcome Distribution")
    plt.show()

# Correlation matrix
px.imshow(df.corr(), title="Correlation Matrix").show()
