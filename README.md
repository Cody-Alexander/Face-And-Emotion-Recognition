# Face-And-Emotion-Recognition
This project uses face recognition and emotion detection to identify and analyze faces in real-time through a webcam. It utilizes the `FER` (Facial Emotion Recognition) library to detect emotions. OpenCV is used for video capture and frame processing.

## Requirements

To run this project, you need to install the following Python packages:

- `opencv-python`
- `FER`
- `tensorflow`

You can install these packages using pip:

```bash
pip install opencv-python FER tensorflow


## Setup
Prepare Known Faces:

Create a directory named known_faces in the project root.
Place image files of known faces in this directory. Each file name should correspond to the person's name (e.g., john_doe.jpg).
Run the Application:

Ensure your webcam is properly connected and accessible.
Execute the main.py script to start the face recognition and emotion detection process.

```bash
python main.py

## Usage:

The application will open a window showing the live feed from your webcam.
Detected faces will be highlighted with rectangles, and their names (if recognized) and emotions will be displayed.
Stopping the Application:

To stop the application, press the 'q' key while the video feed window is active.
Code Overview
main.py
The main script that captures video from the webcam, processes each frame to recognize faces and detect emotions, and displays the results.

face_recognition_util.py
A utility module for loading known faces and recognizing faces and emotions in a given frame.

## Troubleshooting
Camera Issues:

Ensure your camera is properly connected and accessible. Check camera permissions and try different camera indices if necessary.
Installation Issues:

Ensure all required packages are installed. Update or reinstall packages if you encounter errors.
