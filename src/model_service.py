import joblib
import pandas as pd
import logging

model = joblib.load("models/exam_score_model.joblib")
logger = logging.getLogger(__name__)


def predict_exam_score(student_data: dict) -> float:
    logger.info("---- Making prediction -----")
    df = pd.DataFrame([student_data])
    logger.info("Data inserted %s",student_data)
    prediction = model.predict(df)[0]

    logger.info("Prediction Complete %.2f", prediction)
    return prediction
