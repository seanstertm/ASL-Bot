import os
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

# 1. Path to your dataset
DATASET_DIR = "dataset"

# 2. Prepare lists to store data and labels
X_data = []
y_labels = []

# 3. Define a function to preprocess landmarks
def preprocess_landmarks(landmarks):
    """
    Takes a 21x3 NumPy array of MediaPipe landmarks and returns a flattened, normalized array.
    """
    # landmarks shape: (21, 3)
    # Step 1: translate so that the wrist (landmark 0) is at origin
    wrist = landmarks[0]
    landmarks -= wrist
    
    # Step 2: normalize by the maximum absolute value
    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks /= max_val
    
    # Step 3: flatten to shape (63,)
    flattened = landmarks.flatten()
    return flattened

# 4. Initialize MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5
) as hands:

    # 5. Iterate over each folder (A–Z)
    for letter in sorted(os.listdir(DATASET_DIR)):
        letter_folder = os.path.join(DATASET_DIR, letter)
        
        if not os.path.isdir(letter_folder):
            continue
        
        print(f"Processing letter: {letter}")
        
        # 6. Iterate over images in this folder
        for img_name in os.listdir(letter_folder):
            img_path = os.path.join(letter_folder, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Failed to read {img_path}")
                continue
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 7. Process with MediaPipe
            results = hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract the (x, y, z) coordinates
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.append([lm.x, lm.y, lm.z])
                
                landmarks_array = np.array(landmark_list, dtype=np.float32)
                
                # 8. Preprocess and add to dataset
                preprocessed = preprocess_landmarks(landmarks_array)
                X_data.append(preprocessed)
                y_labels.append(letter)
            else:
                # No hands detected
                print(f"No hand found in {img_path}")

# 9. Convert to NumPy arrays
X_data = np.array(X_data)
y_labels = np.array(y_labels)

print(f"Total samples collected: {len(X_data)}")

# 10. Save to a .npz file (or CSV) for training
np.savez("asl_landmarks.npz", X=X_data, y=y_labels)
print("Saved extracted landmarks to asl_landmarks.npz")
