# ASL Robotics
### Developed for TartanHacks'25 by Team SAJE (Sean Richards CS@CMU '27, Andre Miller IS@CMU '27, Jack Fallows ECE+ML@CMU '27, Edelio Taboada ECE+Robotics@CMU '27)

This project demonstrates real-time American Sign Language (ASL) hand-gesture recognition using MediaPipe Hands, a trained classification model (e.g., scikit-learn), and Flask for live video streaming via the web. The code captures frames from a webcam, applies Mediapipe-based hand tracking, predicts ASL letters, and displays the results in a browser. Recognized letters are stored in a text file for later use.

## Features
- Real-Time Hand Detection: Uses MediaPipe Hands to track the hand and extract landmarks in each video frame.
- ASL Classification: A pre-trained model (e.g. scikit-learn’s RandomForest) classifies the extracted 21×3 landmarks into letters A–Z.
- “Hold for 2 Seconds” Logic: Only appends a letter to the recognized text if the same letter is consistently detected for 2 seconds.
- Web Streaming: Streams the annotated camera feed in real time through a Flask web server (using MJPEG).
- Persistence: The recognized letters are saved to recognized_letters.txt whenever the server is stopped or the script exits.
- Robotics: Uses a Raspberry PI 5 to control 5-DOF Robot hand to sign ASL letters based on LLM output

## Table of Contents
- Installation
- Usage
- Running Locally
- Viewing the Stream
- How It Works
- Configuration
- Troubleshooting

## Installation
- Clone this repository:
``` 
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name 
``` 
- Create/Activate a virtual environment (optional but recommended):
``` 
python3 -m venv asl_env
source asl_env/bin/activate  # Linux/Mac
# or .\asl_env\Scripts\activate on Windows
``` 
- Install the required Python packages:
```
pip install -r requirements.txt
```
- Model File: Ensure you have a trained classification model saved as model.pkl in the project folder. This project assumes the model has been trained on ASL hand landmarks and can predict letters A–Z.

## Usage

### Running Locally
- Plug In/Enable your webcam.
- Start the Flask server:
```
python web_stream.py
```
By default, it binds to 0.0.0.0:8000 with debug=False.

- Open a web browser to http://<IP_OF_YOUR_MACHINE>:8000. If you’re on the same machine, http://127.0.0.1:8000 works.

When you’re done, press Ctrl + C in the terminal to stop the server. This triggers a cleanup routine that saves the recognized letters to recognized_letters.txt.

### Viewing the Stream
When you navigate to http://127.0.0.1:8000, you should see:
	•	A simple HTML page with a heading “ASL Live Stream.”
	•	The live video feed from your webcam, with hand landmarks drawn in real time.
	•	As you hold a sign for at least 2 seconds, that letter is appended to the recognized text overlay.

## How It Works
- MediaPipe Hands: In each frame, Mediapipe detects the hand and returns 21 (x, y, z) landmarks.
- Preprocessing: The wrist (landmark 0) is shifted to origin, scaled to normalize distances, and flattened to a 63-dimensional vector (21 × 3).
- Classification: A scikit-learn model (loaded from model.pkl) predicts the most likely letter.
- Hold Time: If the same letter is predicted for 2 seconds, it gets added to a running string recognized_text.
- Overlay: The script draws the current prediction and the recognized letters on the video feed.
- Streaming: Each processed frame is encoded as JPEG and yielded through an MJPEG endpoint. Flask serves these frames at the /video_feed route.
- Final Save: On exit (Ctrl + C), the script writes the recognized text to recognized_letters.txt.

## Configuration
- HOLD_TIME: Change HOLD_TIME = 2.0 to alter how long the user must hold a letter before it’s “locked in.”
- Camera Index: By default, camera = cv2.VideoCapture(0) uses the primary webcam. Change the index if you have multiple cameras or an RTSP/HTTP stream URL.
- Model: If your model is named differently or in a different path, update model = joblib.load("model.pkl").
- Port: The Flask server uses port=8000. You can edit the line app.run(host="0.0.0.0", port=8000) to pick another port.

## Troubleshooting
- No Video: Ensure your webcam is recognized by the OS and not in use by another application.
- Stuck or Black Screen: Try a different index in cv2.VideoCapture(...) or update your camera drivers.
- No Letters Detected: Confirm your model is trained properly and the environment lighting is good for Mediapipe detection.
- Not Saving: Make sure you’re stopping the script with Ctrl + C in the same terminal. If you force-kill the process, the file may never be written.
- High CPU Usage: Real-time inference can be CPU-intensive. Consider using a GPU, or reduce detection confidence thresholds if needed.

## Acknowledgments
- MediaPipe for real-time hand tracking.
- OpenCV for video capture and frame processing.
- Flask for lightweight web serving.
- scikit-learn for machine learning classification.
- TartanHacks'25 for hosting an amazing hackathon

Feel free to open an issue or submit a pull request if you have suggestions, find bugs, or want to add improvements! Enjoy recognizing ASL signs in real-time on the web.
 
