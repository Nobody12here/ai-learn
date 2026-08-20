from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)
student_data = {
    "age": 21,
    "gender": "Female",
    "study_hours_per_day": 5,
    "social_media_hours": 3,
    "netflix_hours": 1.5,
    "part_time_job": "No",
    "attendance_percentage": 95,
    "sleep_hours": 7,
    "diet_quality": "Good",
    "exercise_frequency": 3,
    "parental_education_level": "Bachelor",
    "internet_quality": "Good",
    "mental_health_rating": 9,
    "extracurricular_participation": "Yes",
}


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_prediction():
    response = client.post("/predict", json=student_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "message" in data
    assert "model" in data
    assert data["status"] == "success"


def test_invalid_study_hours():
    data = student_data.copy()
    data["study_hours_per_day"] = 12
    response = client.post("/predict", json=data)

    assert response.status_code == 422


def test_invalid_age():
    data = student_data.copy()
    data["age"] = 100
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_invalid_attendance():
    data = student_data.copy()
    data["attendance_percentage"] = 120
    response = client.post("/predict", json=data)
    assert response.status_code == 422
