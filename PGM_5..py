import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("/content/Healthcare-Diabetes.csv")
#https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes

# Check missing values
print(df.isnull().sum())

# Replace 0 with NaN
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, np.nan)

# Fill missing values with median
for col in cols:
    df[col] = df[col].fillna(df[col].median())

# Create Age Groups
df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[20,30,40,50,60,70,80],
    labels=['21-30','31-40','41-50','51-60','61-70','71-80']
)

# One-hot encoding
df = pd.get_dummies(df, columns=['AgeGroup'], drop_first=True)

# Create interaction feature
df['BMI_Glucose'] = df['BMI'] * df['Glucose']

# Normalize numeric columns
num_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age','BMI_Glucose']

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Create Obese feature
df['Obese'] = (df['BMI'] > 0).astype(int)

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Convert categorical data to numeric
X = pd.get_dummies(X, drop_first=True).astype(float)

print("\nPreprocessing Completed!")
print(X.head())
