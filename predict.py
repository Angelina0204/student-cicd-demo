import joblib


# Load trained model
model = joblib.load("student_placement_model.pkl")


# Student details
student = [[
    8.7,   # CGPA
    88,    # Attendance
    78,    # Coding Score
    3,     # Projects
    1      # Internship
]]


# Make prediction
prediction = model.predict(student)


if prediction[0] == 1:
    print("Student is predicted to be PLACED")
else:
    print("Student is predicted to be NOT PLACED")