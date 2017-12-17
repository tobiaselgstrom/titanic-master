from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

import pandas as pd

X = pd.read_csv("data/train.csv")
y = X.pop("Survived")

X["Age"].fillna(X.Age.mean(), inplace=True)

numeric_variables = list(X.dtypes[X.dtypes != "object"].index)

# print(numeric_variables)
X[numeric_variables].head()

model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=2)

model.fit(X[numeric_variables], y)

y_oob = model.oob_prediction_
print(model.oob_score_)

print("C-stat:", roc_auc_score(y, y_oob))

X.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)


def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"

X["Cabin"] = X.Cabin.apply(clean_cabin)

categorical_variables = ["Sex", "Cabin", "Embarked"]

for variable in categorical_variables:
    X[variable].fillna("Missing", inplace=True)
    dummies = pd.get_dummies(X[variable], prefix=variable)

    X = pd.concat([X, dummies], axis=1)
    X.drop([variable], axis=1, inplace=True)


def printall(X, max_rows=10):
    from IPython.display import display, HTML
    display(HTML(X.to_html(max_rows=max_rows)))

# printall(X)

print(X)

model = RandomForestRegressor(100, oob_score=True, n_jobs=-1, random_state=42)
model.fit(X, y)


y_oob = model.oob_prediction_

#print(y_oob)
print(model.oob_score_)

print("C-stat:", roc_auc_score(y, y_oob))

print(model.feature_importances_)

print("Hej")