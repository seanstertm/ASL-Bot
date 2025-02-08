from flask import Flask, Response
import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
import signal
import sys

app = Flask(__name__)

# 1. Load your model and initialize MediaPipe
model = joblib.load("model.pkl")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 2. Logic for accumulating recognized text
recognized_text = ""
last_letter = None
last_letter_start_time = 0
HOLD_TIME = 2.0

# 3. Open the camera
camera = cv2.VideoCapture(0)

def preprocess_landmarks(landmarks_21x3):
    wrist = landmarks_21x3[0]
    landmarks_21x3 -= wrist
    max_val = np.max(np.abs(landmarks_21x3))
    if max_val > 0:
        landmarks_21x3 /= max_val
    return landmarks_21x3.flatten()

def generate_frames():
    global recognized_text, last_letter, last_letter_start_time
    
    while True:
        success, frame = camera.read()
        if not success:
            break

        # 4. Flip frame + convert color for MediaPipe
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 5. Run MediaPipe hand detection
        results = hands.process(rgb_frame)

        # 6. If we find a hand, classify the letter
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks for visualization
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract (x,y,z) from each landmark
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.append([lm.x, lm.y, lm.z])
                landmarks_21x3 = np.array(landmark_list, dtype=np.float32)

                # Preprocess + predict the letter
                features = preprocess_landmarks(landmarks_21x3)
                predicted_letter = model.predict([features])[0]

                # 7. Only add the letter if held for 2 seconds
                if predicted_letter == last_letter:
                    elapsed = time.time() - last_letter_start_time
                    if elapsed >= HOLD_TIME:
                        recognized_text += predicted_letter
                        print(f"Added letter: {predicted_letter} => {recognized_text}")
                        last_letter = None
                else:
                    last_letter = predicted_letter
                    last_letter_start_time = time.time()

                # Display the current prediction on the frame
                cv2.putText(frame, f"Prediction: {predicted_letter}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

        # Show the recognized text so far
        cv2.putText(frame, f"Recognized: {recognized_text}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)

        # 8. Encode the processed frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # 9. Yield the frame in multipart boundary format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """
    Endpoint that provides the streaming video feed.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """
    A simple page that shows the live video stream in an <img> tag.
    """
    return '''
    <html>
      <head><title>ASL Video Stream</title></head>
      <body>
        <h1>ASL Live Stream</h1>
        <img src="/video_feed" width="640" height="480" />
      </body>
    </html>
    '''

def cleanup_and_save():
    """
    Clean up camera and save recognized_text on exit.
    """
    camera.release()
    cv2.destroyAllWindows()
    with open("recognized_letters.txt", "w") as f:
        f.write(recognized_text)
    print("Letters saved to recognized_letters.txt")

def signal_handler(sig, frame):
    cleanup_and_save()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)  # handle Ctrl-C
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    # 10. Run Flask
    try:
        app.run(host="0.0.0.0", port=8000, debug=False)
    finally:
        cleanup_and_save()