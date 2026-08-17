from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI(debug=True)
model = joblib.load("models/exam_score_model.joblib")


class StudentModel(BaseModel):
    age: int
    gender: str
    study_hours_per_day: float
    social_media_hours: float
    netflix_hours: float
    part_time_job: str
    attendance_percentage: float
    sleep_hours: float
    diet_quality: str
    exercise_frequency: int
    parental_education_level: str
    internet_quality: str
    mental_health_rating: int
    extracurricular_participation: str


@app.get("/")
def home():
    return {"message": "Student Score ML API running..."}


@app.post("/predict")
def predict(student: StudentModel):
    student_df = pd.DataFrame([student.model_dump()])

    prediction = model.predict(student_df)
    return {"message": f"Exam score {prediction[0]}", "data": student}
