import cv2
import numpy as np
import mediapipe as mp
import joblib
import time

model = joblib.load("model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def preprocess_landmarks(landmarks_21x3):
    wrist = landmarks_21x3[0]
    landmarks_21x3 -= wrist
    max_val = np.max(np.abs(landmarks_21x3))
    if max_val > 0:
        landmarks_21x3 /= max_val
    return landmarks_21x3.flatten()

recognized_text = ""
last_letter = None
last_letter_start_time = 0
HOLD_TIME = 2.0

cap = cv2.VideoCapture(0)

try:
    while True:
        success, frame = cap.read()
        if not success:
            continue
    
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.append([lm.x, lm.y, lm.z])
                landmarks_21x3 = np.array(landmark_list, dtype=np.float32)
                
                features = preprocess_landmarks(landmarks_21x3)
                predicted_letter = model.predict([features])[0]
                
                if predicted_letter == last_letter:
                    elapsed = time.time() - last_letter_start_time
                    if elapsed >= HOLD_TIME:
                        recognized_text += predicted_letter
                        print(f"Added letter: {predicted_letter} => {recognized_text}")
                        last_letter = None
                else:
                    last_letter = predicted_letter
                    last_letter_start_time = time.time()
                
                cv2.putText(frame, f"Prediction: {predicted_letter}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
        
        # Show the recognized text so far
        cv2.putText(frame, f"Recognized: {recognized_text}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)
        
        cv2.imshow("ASL Live Demo", frame)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

# --- AFTER THE LOOP EXITS, SAVE recognized_text TO A FILE ---
with open("recognized_letters.txt", "w") as f:
    f.write(recognized_text)

print("Letters saved to recognized_letters.txt")