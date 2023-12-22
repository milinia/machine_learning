import os

import cv2
import numpy as np
from PIL import Image


def create_dataset():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    count = 0
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite("faceDataset/evelina" + str(count) + ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('Face ID', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= 300:
            break

    cap.release()
    cv2.destroyAllWindows()

def test_model():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("model.yml")
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 50:
                put_text_on_video("Evelina", frame, x, y)
            else:
                put_text_on_video("Unknown", frame, x, y)

        cv2.imshow('Face ID', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def put_text_on_video(text, frame, x, y):
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

def train_model(path):
    image_path = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    labels = []
    for path in image_path:
        image = Image.open(path).convert('L')
        faces.append(np.array(image, 'uint8'))
        labels.append(1)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = np.array(labels)
    recognizer.train(faces, labels)
    recognizer.write("model.yml")

if __name__ == '__main__':
    # create_dataset()
    # train_model("faceDataset")
    test_model()