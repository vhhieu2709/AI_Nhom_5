from asl_model import ASLRecognizer


if __name__ == "__main__":
    recognizer = ASLRecognizer(model_path="model_baseline.pkl", hand_task_path="hand_landmarker.task")
    recognizer.predict_webcam()
