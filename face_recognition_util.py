import cv2
import os
from fer import FER
import numpy as np

# initialize the emotion detector
emotion_detector = FER()

# function to load known faces from images
def load_known_faces():
    known_faces_dir = "known_faces"
    known_faces = []
    known_names = []

    # loop through each image in the directory
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(known_faces_dir, filename)
            image = cv2.imread(image_path)

            # convert the image to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # create a face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # store each face found in the image
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                known_faces.append(face_img)
                known_names.append(filename.split('.')[0])

    return known_faces, known_names

# function to recognize faces and detect emotions in a frame
def recognize_face(frame, known_faces, known_names):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # create a face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    face_names_emotions = []

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_region = frame[y:y+h, x:x+w]

        # compare with known faces
        min_distance = float('inf')
        name = "Unknown"
        for known_face, known_name in zip(known_faces, known_names):
            # calculate the mean squared error for the face comparison
            distance = np.sum((face_img - known_face) ** 2)
            if distance < min_distance:
                min_distance = distance
                name = known_name

        # detect emotion
        emotion, _ = emotion_detector.top_emotion(face_region)

        # add the face name and detected emotion to the list
        face_names_emotions.append(((x, y, x+w, y+h), name, emotion))

    return face_names_emotions
