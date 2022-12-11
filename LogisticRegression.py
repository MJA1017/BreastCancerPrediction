#%% Loading Data
from dataset.classificationData import X, y, X_train, X_test, y_train, y_test

# Load Logistic Regression model from SkLearn
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)
model.score(X_test, y_test)