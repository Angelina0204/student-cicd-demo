import os
import joblib
import pandas as pd


def test_dataset_exists():
    assert os.path.exists("data/student_placement.csv")


def test_model_file_exists():
    assert os.path.exists("student_placement_model.pkl")


def test_model_prediction():
    model = joblib.load("student_placement_model.pkl")

    student = pd.DataFrame([{
        "CGPA": 8.2,
        "Attendance": 88,
        "CodingScore": 78,
        "Projects": 3,
        "Internship": 1
    }])

    prediction = model.predict(student)

    assert prediction[0] in [0, 1]