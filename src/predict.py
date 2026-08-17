import joblib
import pandas as pd

model = joblib.load("models/exam_score_model.joblib")
preprocessor = model.named_steps["preprocess"]
featurs = preprocessor.get_feature_names_out()
linear_model = model.named_steps["model"]
coeff = pd.DataFrame({"feature": featurs, "coefficient": linear_model.coef_})
print(linear_model.coef_)
print(coeff.sort_values("coefficient",ascending=False).head(50))

student = {
    "age": 21,
    "gender": "Male",
    "study_hours_per_day": 8.0,
    "social_media_hours": 2.0,
    "netflix_hours": 1.5,
    "part_time_job": "No",
    "attendance_percentage": 85.0,
    "sleep_hours": 7.0,
    "diet_quality": "Good",
    "exercise_frequency": 3,
    "parental_education_level": "Bachelor",
    "internet_quality": "Good",
    "mental_health_rating": 7,
    "extracurricular_participation": "Yes",
}
student_df = pd.DataFrame([student])
student.update(study_hours_per_day=2.0)
student_df2 = pd.DataFrame([student])
prediction_more_study_hours = model.predict(student_df)
prediction_less_study_hours = model.predict(student_df2)
# print("Exam score of student with 8 hours of study :", prediction_more_study_hours[0])
# print("Exam score of student with 2 hours of study :", prediction_less_study_hours[0])
