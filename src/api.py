from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, Field
from .model_service import predict_exam_score
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
app = FastAPI(debug=True)


class StudentModel(BaseModel):
    age: int = Field(ge=17, le=25)
    gender: str
    study_hours_per_day: float = Field(ge=0, le=8.3)
    social_media_hours: float
    netflix_hours: float
    part_time_job: str
    attendance_percentage: float = Field(ge=0, le=100)
    sleep_hours: float = Field(ge=0, le=24)
    diet_quality: str
    exercise_frequency: int
    parental_education_level: str
    internet_quality: str
    mental_health_rating: int = Field(ge=1, le=10)
    extracurricular_participation: str


@app.get("/")
def home():
    return {"message": "Student Score ML API running..."}


@app.post("/predict")
def predict(student: StudentModel):
    logger.info("Making prediction request")
    prediction = predict_exam_score(student.model_dump())
    if prediction < 0 or prediction > 100:
        logger.error("Invalid model prediction .%2f", prediction)
        raise HTTPException(status_code=500, detail="Error occured when prediction")
    logger.info("prediction sucessfull %.2f", prediction)
    return {
        "prediction": round(float(prediction), 3),
        "model": "exam_score_model",
        "message": "Scores predicted sucessfully",
        "status": "success",
    }
