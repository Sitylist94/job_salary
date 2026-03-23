import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import lightgbm as lgb
# from pycaret.regression import *

data = pd.read_csv("../data/job_salary_prediction_dataset.csv")
df = data.copy()

X = df.drop(["salary"], axis=1)
y = df["salary"]

nums_col = ["experience_years", "skills_count", "certifications"]
cats_col = ["job_title", "education_level", "industry", "company_size", "location", "remote_work"]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="mean"), nums_col),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ]), cats_col)

])

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", lgb.LGBMRegressor())
])

# pipe.fit(X, y)

scores = cross_val_score(pipe, X, y, cv=10)
print(scores.mean())

# y_pred = pipe.predict(X)
# print(r2_score(y_pred, y))

# setup(data=df, target="salary", session_id=42)
# model = create_model("lightgbm")
# tunned_model = tune_model(model)
# final = finalize_model(tunned_model)

# final.get_params()
