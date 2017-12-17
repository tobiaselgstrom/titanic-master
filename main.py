from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
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

model = RandomForestRegressor(100, oob_score=True, n_jobs=-1, random_state=42)
model.fit(X, y)


y_oob = model.oob_prediction_

#print(y_oob)
print(model.oob_score_)

print("C-stat:", roc_auc_score(y, y_oob))

"""
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values()

t = feature_importances.plot(kind="barh")
plt.show(t)

results = []
n_estimator_options = [30, 50, 100, 200, 500, 1000, 2000]

for trees in n_estimator_options:
    model = RandomForestRegressor(trees, oob_score=True, random_state=42)
    model.fit(X, y)
    print(trees, "trees.")
    roc = roc_auc_score(y, model.oob_prediction_)
    print("C-stat:", roc)
    results.append(roc)
    print("")

t1 = pd.Series(results, n_estimator_options).plot()
plt.show(t1)

results = []
max_features_options = ["auto", None, "sqrt", "log2", 0.9, 0.2]

for max_features in max_features_options:
    model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=42, max_features=max_features)
    model.fit(X, y)
    print(max_features, "option.")
    roc = roc_auc_score(y, model.oob_prediction_)
    print("C-stat:", roc)
    results.append(roc)
    print("")


t2 = pd.Series(results, max_features_options).plot(kind="barh", xlim=(0.85, 0.88))
plt.show(t2)
"""
model = RandomForestRegressor(n_estimators=1000,
                              oob_score=True,
                              n_jobs=-1,
                              random_state=42,
                              max_features="auto",
                              min_samples_leaf=5)

model.fit(X, y)
roc = roc_auc_score(y, model.oob_prediction_)
print("C-stat:", roc)

y_oob = model.oob_prediction_
print(y_oob)

