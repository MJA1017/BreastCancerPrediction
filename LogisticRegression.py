#%% Loading Data

import pandas as pd

filename = "top10features.csv"

df = pd.read_csv(filename)
df.info()


#%% Remove first column of df

df.drop(df.columns[0], axis=1, inplace=True)
df


#%% Add the Y value to the dataframe
predictor = pd.read_csv('cleaned.csv')
temp = pd.DataFrame(predictor.iloc[: , [1]])

# Append the Y value
df['outcome'] = temp
df

#%% Outcomes
effective = df[df.outcome == 1]
not_effective = df[df.outcome == 0]

x = df.iloc[:, 0:10]
y = df.iloc[:, [-1]]

#%% Preparing the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 8675309)

# Load Logistic Regression model from SkLearn
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)
model.score(X_test,y_test)
