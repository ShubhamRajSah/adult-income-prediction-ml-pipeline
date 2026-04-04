import pandas as pd
df = pd.read_csv("adult.csv", na_values="?")

print("------HEAD------")
print(df.head())
print("------INFORMATION------")
print(df.info())
print("------DESCRIBE------")
print(df.describe())
print("------NUMBER OF MISSING IN EACH COLUMN------")
print(df.isnull().sum())

df = df.drop("fnlwgt", axis=1)

df['income'] = df["income"].str.strip()

df["workclass"].fillna("Missing", inplace=True)
df["occupation"].fillna("Missing", inplace=True)
df["native-country"].fillna("Missing", inplace=True)

print(df.isnull().sum())

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
    ('imputer', SimpleImputer(strategy='most_frequent')),
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
    ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
])


print("--------------cross validation---------------")
# from sklearn.model_selection import cross_val_score   

# scores_li = cross_val_score(
#     model_pipeline, 
#     x_train, 
#     y_train, 
#     cv=5, 
#     scoring='f1'
# )
# print("linear model")
# print("Cross-validation scores:", scores_li)
# print("Mean f1score:", scores_li.mean())
# print("Std deviation:", scores_li.std())

# using gridsearchcv for hyperparameter optimization
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__C': [0.01, 0.1, 1, 10],
    'model__l1_ratio': [0, 1],
    'model__solver': ['liblinear']
}

grid = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid.fit(x_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)
best_model = grid.best_estimator_

probs = best_model.predict_proba(x_test)[:, 1]
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# for threshold in [0.3, 0.5, 0.7]:
#     y_pred = (probs >= threshold).astype(int)

#     print(f"\nThreshold = {threshold}")
#     print(confusion_matrix(y_test, y_pred))
#     print(classification_report(y_test, y_pred))

from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# precision, recall, thresholds = precision_recall_curve(y_test, probs)

# plt.figure(figsize=(8,5))
# plt.plot(thresholds, precision[:-1], label='Precision')
# plt.plot(thresholds, recall[:-1], label='Recall')
# plt.xlabel("Threshold")
# plt.ylabel("Score")
# plt.title("Precision and Recall vs Threshold")
# plt.legend()
# plt.grid(True)
# plt.show()
print("succesfully printed plot graph.")
fpr, tpr, _ = roc_curve(y_test, probs)
auc_score = roc_auc_score(y_test, probs)
print("ROC-AUC:", auc_score)

plt.figure(figsize=(8,5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

print("succesfully printed roc graph.")

# model_pipeline.fit(x_train, y_train)

# from sklearn.metrics import classification_report, confusion_matrix

# y_pred = model_pipeline.predict(x_test)
# print("------------LINEAR REGRESSION--------------")
# print('CONFUSION MATRIX\n', confusion_matrix(y_test, y_pred))
# print("CLASSIFICATION REPORT\n", classification_report(y_test, y_pred))


# # print("----------RANDOM FOREST CLASSIFIER-----------")
from sklearn.ensemble import RandomForestClassifier

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])


# scores_rf = cross_val_score(
#     rf_pipeline,
#     x_train,
#     y_train,
#     cv=5,
#     scoring="f1"
# )
# print("random forest")
# print("Cross-validation scores:", scores_rf)
# print("Mean f1score:", scores_rf.mean())
# print("Std deviation:", scores_rf.std())


# rf_pipeline.fit(x_train, y_train)

# y_pred_rf = rf_pipeline.predict(x_test)

# print(confusion_matrix(y_test, y_pred_rf))
# print(classification_report(y_test, y_pred_rf))


# print("--------------THRESHOLD TUNING----------------")


# import numpy as np
# y_prob = rf_pipeline.predict_proba(x_test)[:, 1]

# # Try 0.3
# y_pred_03 = (y_prob > 0.3).astype(int)

# # Try 0.7
# y_pred_07 = (y_prob > 0.7).astype(int)

# print("Threshold 0.3")
# print(confusion_matrix(y_test, y_pred_03))
# print(classification_report(y_test, y_pred_03))

# print("\nThreshold 0.7")
# print(confusion_matrix(y_test, y_pred_07))
# print(classification_report(y_test, y_pred_07))


# print("---------------FEATURE IMPORTANCE----------------")


# rf_model = rf_pipeline.named_steps["model"]
# preprocessor = rf_pipeline.named_steps["preprocessor"]

# cat_features = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_cols)

# all_features = list(num_cols) + list(cat_features)

# importance = rf_model.feature_importances_

# feature_importance_df = pd.DataFrame({
#     "feature": all_features,
#     "importance": importance
# }).sort_values(by="importance", ascending=False)

# # print(feature_importance_df.head(10))
# for i in range(10):
#     print(i, all_features[i])
# print(all_features[32])


# print("----------------SHAP EXPLAINABILITY------------------")

# # import shap

# # # transform the test data, the result of transformed data will be in sparse matrix form but the shap requires full -
# # # - numeric data and also float. also using only first 200 data cause it will take more time if all 48842 rows will be used for shap - 
# # # - as it will have to go through all the deep tree. 


# # # ✅ Step 1: Transform test data
# # x_test_transformed = preprocessor.transform(x_test)
# # X_sample = x_test_transformed.toarray().astype(np.float64)[:100]

# # explainer = shap.TreeExplainer(rf_model)
# # shap_values = explainer(X_sample)

# # # using beeswarm to avoid interaction.
# # # ✅ Select class 1
# # shap_values_class1 = shap_values[:, :, 1]  #making one dimension as beeswarm doesn't work well on multiple dimensions.

# # # ✅ Plot
# # shap.plots.beeswarm(shap_values_class1)
# # print("plot done.")
# # # Summary plot
# # shap.summary_plot(shap_values, X_sample, feature_names=all_features)
# # print("Summary plot done.")
# # # Force plot
# # shap.initjs()
# # shap.force_plot(
# #     explainer.expected_value,
# #     shap_values[0],
# #     X_sample[0],
# #     feature_names=all_features
# # )
# # print("force plot done.")

