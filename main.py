from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

import pandas as pd

X = pd.read_csv("data/train.csv")
y = X.pop("Survived")

X["Age"].fillna(X.Age.mean(), inplace=True)
# X.describe()

numeric_variables = list(X.dtypes[X.dtypes != "object"].index)

# print(numeric_variables)
X[numeric_variables].head()

model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)

model.fit(X[numeric_variables], y)

print(model.oob_score_)


