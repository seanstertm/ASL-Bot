import cv2
import numpy as np
import mediapipe as mp

# For drawing the landmarks on the image (optional, for visualization)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Example classifier placeholder (replace with your trained model)
class ASLClassifier:
    def __init__(self):
        # Load or initialize your model here
        # E.g., self.model = load_model('my_asl_model.h5') if using TensorFlow
        pass
    
    def predict(self, landmarks_array):
        """
        Takes a normalized, flattened landmarks array and 
        returns a predicted letter (A, B, C, ...).
        """
        # In practice: predictions = self.model.predict([landmarks_array])
        # predicted_label = np.argmax(predictions[0])
        # return label_encoder.inverse_transform([predicted_label])[0]
        
        # For demo, return a placeholder letter
        return "?"

def main():
    # Initialize video capture (webcam)
    cap = cv2.VideoCapture(0)
    
    # Create an instance of our classifier
    classifier = ASLClassifier()
    
    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,          # False -> detection + tracking in video
        max_num_hands=1,                 # Detect up to 2 hands
        model_complexity=1,              # Model complexity (0 or 1)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while True:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Flip the frame horizontally for natural selfie-view
            frame = cv2.flip(frame, 1)
            
            # Convert from BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame to find hands/landmarks
            results = hands.process(rgb_frame)
            
            # Draw the hand annotations on the frame (optional)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Extract the raw landmarks
                    landmark_list = []
                    for lm in hand_landmarks.landmark:
                        landmark_list.append([lm.x, lm.y, lm.z])
                    
                    # Preprocess the landmarks for classification
                    landmarks_array = preprocess_landmarks(landmark_list)
                    
                    # Classify the hand gesture
                    predicted_letter = classifier.predict(landmarks_array)
                    
                    # Visualize the predicted letter
                    cv2.putText(
                        frame, 
                        f"Letter: {predicted_letter}", 
                        (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
            
            cv2.imshow("ASL Hand Gesture Recognition", frame)
            
            # Press 'q' to exit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

def preprocess_landmarks(landmark_list):
    """
    Normalizes and flattens the 21 x 3 landmarks into a 1D vector.
    
    Steps often include:
    1. Translate coordinates so that the wrist (landmark 0) is at origin.
    2. Scale/normalize the coordinates for consistency.
    3. Flatten into 1D array for model input.
    """
    # Convert to NumPy for easier manipulation
    landmarks = np.array(landmark_list)
    
    # Step 1: Translate so that the wrist (index 0) is at (0,0,0)
    wrist = landmarks[0]
    landmarks -= wrist
    
    # Step 2: Optional - scale by max distance to remove size variability
    max_value = np.max(np.abs(landmarks))
    if max_value > 0:
        landmarks /= max_value
    
    # Step 3: Flatten
    flattened = landmarks.flatten()
    
    return flattened

if __name__ == "__main__":
    main()