import pandas as pd
df = pd.read_csv("adult.csv", na_values="?")

df = df.drop("fnlwgt", axis=1)

df['income'] = df["income"].str.strip()

df["workclass"].fillna("Missing", inplace=True)
df["occupation"].fillna("Missing", inplace=True)
df["native-country"].fillna("Missing", inplace=True)

df['age_group'] = pd.cut(df['age'],
                         bins=[0, 25, 40, 60, 100],
                         labels= ["Young", "Adult", "Mid", "Senior"])




df['hours_group'] = pd.cut(df['hours-per-week'],
                           bins=[0, 25, 40, 60, 100],
                           labels=["Part-time", "Normal", "Overtime", "Heavy"])



df["native-country"] = df['native-country'].apply(
    lambda x: "US" if x == 'United-States' else 'Other'
)

df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

x = df.drop('income', axis =1 )
y = df['income']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

num_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = [cols for cols in x.columns if cols not in num_cols]

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])


from sklearn.linear_model import LogisticRegression


model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        l1_ratio = 1,  # 1 for lasso and 2 for ridge
        C=0.1,
        solver='liblinear',
        random_state=42
    ))
])

model_pipeline.fit(x_train, y_train)

from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    classification_report,
    confusion_matrix
)

y_prob = model_pipeline.predict_proba(x_test)[:, 1]
threshold = 0.7
y_pred = (y_prob >= threshold).astype(int)

print("F1:", f1_score(y_test, y_pred))
print('accuracy score:',accuracy_score(y_test, y_pred) )
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))


import joblib
final_threshold = 0.7
joblib.dump({
    "model": model_pipeline,
    "threshold": final_threshold
}, "adult_income_final.pkl")

