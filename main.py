from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

import pandas as pd

X = pd.read_csv("data/train.csv")
y = X.pop("Survived")

