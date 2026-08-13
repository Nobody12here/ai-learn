import pandas as pd

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

df = pd.read_csv("data/student_habits_performance.csv")
x = df.drop(columns=["student_id", "exam_score"])
Y = df["exam_score"]

x_train, x_test, y_train, y_test = train_test_split(
    x, Y, test_size=0.2, random_state=32
)

numerical_features = [
    "age",
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "attendance_percentage",
    "sleep_hours",
    "exercise_frequency",
    "mental_health_rating",
]
categorical_features = [
    "gender",
    "part_time_job",
    "diet_quality",
    "extracurricular_participation",
    "internet_quality",
    "parental_education_level",
]
numerical_transformer = StandardScaler()
categorical_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    [
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        ("model", LinearRegression()),
    ]
)
model_pipeline.fit(x_train, y_train)
joblib.dump(model_pipeline,"models/exam_score_model.joblib")
print("Model saved sucessfully....")
# rf = model_pipeline.named_steps["model"]
# feature_names = model_pipeline.named_steps["preprocess"].get_feature_names_out()
# importance = (
#     pd.DataFrame({"feature": feature_names, "Importance": rf.feature_importances_})
#     .sort_values("Importance", ascending=False)
#     .head(10)
# )

# param_grid = {
#     "model__n_estimators": [50, 100, 200],
#     "model__max_depth": [None, 5, 10, 20],
# }
# grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
# grid_search.fit(x_train,y_train)
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(x_test)


# print("best parameter = ",grid_search.best_params_)
# print("best CV r2 = ",grid_search.best_score_)


# score = cross_val_score(model_pipeline, x_train, y_train, cv=5, scoring="r2")
# print("Cross Value scores", score)
# print("Average R2 ", score.mean())
y_pred = model_pipeline.predict(x_test)

# results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
# print(results.head(10))
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MAE = ", mae)
print("RMSE = ", rmse)
print("R2 = ", r2)
