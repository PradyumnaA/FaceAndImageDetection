import cv2
import os
import numpy as np

# Initialize face recognizer and Haar cascade
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load and train face data
def load_faces(train_dir='assets/images/'):
    images, labels = [], []
    label_names = {}
    name_to_id = {}
    current_id = 0

    for filename in os.listdir(train_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(train_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            name = os.path.splitext(filename)[0].split('_')[0]
            if name not in name_to_id:
                name_to_id[name] = current_id
                label_names[current_id] = name
                current_id += 1
            images.append(cv2.resize(img, (200, 200)))
            labels.append(name_to_id[name])

    face_recognizer.train(images, np.array(labels))
    return label_names

# Recognize faces in the frame
def recognize_person(frame, label_names):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    recognized = []

    for (x, y, w, h) in faces:
        roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        label, confidence = face_recognizer.predict(roi)
        if confidence < 100:
            name = label_names.get(label, "Undefined")
        else:
            name = "Undefined"
        recognized.append((name, (x, y, x+w, y+h)))

    return recognized
