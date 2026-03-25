# Healthcare EDA Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('diabetes.csv')

# Data Understanding
print(df.shape)
print(df.info())
print(df.describe())

# Data Cleaning
cols = ['Glucose','BloodPressure','BMI','Insulin']
df[cols] = df[cols].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

# Univariate Analysis
df.hist(figsize=(10,8))
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Target Analysis
sns.countplot(x='Outcome', data=df)
plt.show()

# Feature Comparison
sns.boxplot(x='Outcome', y='Glucose', data=df)
plt.show()

sns.boxplot(x='Outcome', y='BMI', data=df)
plt.show()

# Pairplot
sns.pairplot(df, hue='Outcome')
plt.show()
