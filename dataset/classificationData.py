# Files to import variables
import pandas as pd

filename = "top10features.csv"

df = pd.read_csv(filename)

df.drop(df.columns[0], axis=1, inplace=True)

#%% Add the Y value to the dataframe
predictor = pd.read_csv('cleaned.csv')
temp = pd.DataFrame(predictor.iloc[: , [1]])

# Append the Y value
df['outcome'] = temp

#%% Outcomes
import numpy as np

effective = df[df.outcome == 1]
not_effective = df[df.outcome == 0]

X = df.iloc[:, 0:10]
y = df.iloc[:, [-1]]
y = np.ravel(y)

#%% Preparing the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 8675309)

df.info()