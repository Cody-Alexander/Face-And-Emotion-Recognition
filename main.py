import cv2
from face_recognition_util import load_known_faces, recognize_face

known_faces, known_names = load_known_faces()

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    face_names_emotions = recognize_face(frame, known_faces, known_names)

    for (top, right, bottom, left), name, emotion in face_names_emotions:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 50), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, f"{name}: {emotion}", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
