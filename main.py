import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image, ImageFormat

# ----------------------------
# IMAGE FOLDER PATH
# ----------------------------
# Update this to your new images folder
IMAGE_DIR = r"C:\Users\BHAVYA\Downloads\mimi_project"

print("IMAGE DIRECTORY:", IMAGE_DIR)
print("FILES FOUND:", os.listdir(IMAGE_DIR))

def load_image(name):
    path = os.path.join(IMAGE_DIR, name)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Keep alpha channel if PNG
    if img is None:
        print(f"ERROR: {name} not loaded. Check filename or extension!")
    else:
        print(f"SUCCESS: {name} loaded successfully")
    return img

# Load images
hand_in_nose_img = load_image("hand_in_nose.png")
hands_on_head_img = load_image("hands_on_head.png")
shock_speed_img = load_image("shock_speed.png")
smile_wide_img = load_image("smile_wide.png")
thinking_monkey_img = load_image("thinking_moneky.png")

# Stop if any image failed
if hand_in_nose_img is None or hands_on_head_img is None or shock_speed_img is None or smile_wide_img is None or thinking_monkey_img is None:
    print("ERROR: Images not loaded. Check filenames and location.")
    exit()

# ----------------------------
# MODEL DOWNLOAD AND SETUP
# ----------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

face_model_path = os.path.join(MODEL_DIR, 'face_landmarker.task')
hand_model_path = os.path.join(MODEL_DIR, 'hand_landmarker.task')

def download_model(url, path):
    if not os.path.exists(path):
        print(f"Downloading {url} to {path}")
        urllib.request.urlretrieve(url, path)

download_model('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', face_model_path)
download_model('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', hand_model_path)

face_base_options = mp_tasks.BaseOptions(model_asset_path=face_model_path)
face_options = mp_vision.FaceLandmarkerOptions(
    base_options=face_base_options, 
    output_face_blendshapes=True, 
    output_facial_transformation_matrixes=True, 
    num_faces=1
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)

hand_base_options = mp_tasks.BaseOptions(model_asset_path=hand_model_path)
hand_options = mp_vision.HandLandmarkerOptions(
    base_options=hand_base_options, 
    num_hands=2
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

# ----------------------------
# CAMERA SETUP
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

# Optimize camera settings for speed and quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frame

# Get frame size for video writer
ret, test_frame = cap.read()
if ret:
    h, w, _ = test_frame.shape
else:
    w, h = 1280, 720  # fallback

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('mimi_output.avi', fourcc, 30.0, (w, h))

# Performance tracking
frame_count = 0
start_time = time.time()

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def hand_in_nose(face_landmarks, hand_landmarks, h, w):
    """Check if hand index finger is near nose"""
    nose = face_landmarks[4]  # nose tip
    nose_x = nose.x
    nose_y = nose.y
    
    index_tip = hand_landmarks[8]  # index finger tip
    finger_x = index_tip.x
    finger_y = index_tip.y
    
    # Distance threshold (normalized coordinates)
    distance = np.sqrt((nose_x - finger_x)**2 + (nose_y - finger_y)**2)
    return distance < 0.145 and finger_y < 0.50

def hands_on_head(hand_landmarks_list, face_landmarks, h, w):
    """Check if both hands are above head"""
    if len(hand_landmarks_list) < 2:
        return False
    
    nose = face_landmarks[1]  # nose position
    nose_y = nose.y
    
    hands_above = 0
    for hand_landmarks in hand_landmarks_list:
        wrist = hand_landmarks[0]
        wrist_y = wrist.y
        # Check if wrist is above nose (head area) - stricter threshold
        if wrist_y < nose_y - 0.15 and wrist_y > 0.05:  # Avoid top edge noise
            hands_above += 1
    
    return hands_above >= 2

def shock_expression(face_landmarks, h):
    """Check for shocked expression (wide mouth and eyes)"""
    # Check mouth opening
    upper_lip = face_landmarks[13]
    lower_lip = face_landmarks[14]
    mouth_distance = abs((lower_lip.y - upper_lip.y))
    
    # Check eye opening (both eyes)
    left_eye_top = face_landmarks[159]
    left_eye_bottom = face_landmarks[145]
    right_eye_top = face_landmarks[386]
    right_eye_bottom = face_landmarks[374]
    
    left_eye_distance = abs((left_eye_bottom.y - left_eye_top.y))
    right_eye_distance = abs((right_eye_bottom.y - right_eye_top.y))
    
    return mouth_distance > 0.045 and (left_eye_distance > 0.010 and right_eye_distance > 0.010)

def smile_wide(face_landmarks):
    """Check for wide smile using mouth corner distance and position"""
    # Mouth corners
    left_mouth_corner = face_landmarks[57]
    right_mouth_corner = face_landmarks[287]
    mouth_width = abs(right_mouth_corner.x - left_mouth_corner.x)
    
    # Mouth center points for smile detection
    upper_lip = face_landmarks[13]
    lower_lip = face_landmarks[14]
    mouth_opening = abs((lower_lip.y - upper_lip.y))
    
    # Check if corners are raised (smile) and mouth is wide
    corners_raised = (left_mouth_corner.y < 0.58 and right_mouth_corner.y < 0.58)
    return mouth_width > 0.078 and mouth_opening > 0.02 and corners_raised

def thinking_pose(hand_landmarks_list, face_landmarks, h, w):
    """Check if hand is near chin (thinking pose - hand at chin level)"""
    if not hand_landmarks_list:
        return False
    
    # Chin position (lower face)
    chin = face_landmarks[152]
    chin_x = chin.x
    chin_y = chin.y
    
    for hand_landmarks in hand_landmarks_list:
        # Check hand palm and fingers near chin only
        for point_idx in [5, 9, 12]:  # palm points and middle finger
            finger = hand_landmarks[point_idx]
            finger_x = finger.x
            finger_y = finger.y
            
            # Distance in normalized coordinates - must be close to chin
            distance = np.sqrt((chin_x - finger_x)**2 + (chin_y - finger_y)**2)
            # Only trigger if at lower face level (below middle of face) - stricter bounds
            if distance < 0.13 and finger_y > 0.50 and finger_y < 0.85:
                return True
    return False

def overlay_image_alpha(background, overlay, x, y, size=(280,280)):
    """Overlay transparent PNG on BGR frame"""
    overlay = cv2.resize(overlay, size)
    
    # Ensure x, y are within bounds
    x = max(0, min(x, background.shape[1] - size[0]))
    y = max(0, min(y, background.shape[0] - size[1]))
    
    if overlay.shape[2] == 4:  # has alpha channel
        alpha = overlay[:, :, 3] / 255.0
        for c in range(0, 3):
            background[y:y+size[1], x:x+size[0], c] = (
                alpha * overlay[:, :, c] + (1-alpha) * background[y:y+size[1], x:x+size[0], c]
            )
    else:
        background[y:y+size[1], x:x+size[0]] = overlay
    return background

# ----------------------------
# MAIN LOOP
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w, _ = frame.shape
    
    # Convert to RGB for MediaPipe (faster without copy)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

    # Run detection in parallel (async in LIVE_STREAM mode)
    face_result = face_landmarker.detect(mp_image)
    hand_result = hand_landmarker.detect(mp_image)

    overlay = None

    if face_result.face_landmarks:
        face_landmarks = face_result.face_landmarks[0]
        
        if hand_result.hand_landmarks:
            hand_landmarks_list = hand_result.hand_landmarks
            
            # Check for hands on head (priority 1)
            if hands_on_head(hand_landmarks_list, face_landmarks, h, w):
                overlay = hands_on_head_img
            # Check for thinking pose (priority 2)
            elif thinking_pose(hand_landmarks_list, face_landmarks, h, w):
                overlay = thinking_monkey_img
            # Check for hand in nose (priority 3)
            elif hand_in_nose(face_landmarks, hand_landmarks_list[0], h, w):
                overlay = hand_in_nose_img
        
        # Check for shock expression (priority 4)
        if overlay is None and shock_expression(face_landmarks, h):
            overlay = shock_speed_img
        # Check for smile (priority 5)
        elif overlay is None and smile_wide(face_landmarks):
            overlay = smile_wide_img

    if overlay is not None:
        frame = overlay_image_alpha(frame, overlay, w//2, 0, size=(w//2, h))

    # Display FPS
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        print(f"FPS: {fps:.1f}")

    out.write(frame)
    cv2.imshow("Mimi Project 🐵", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
out.release()
cv2.destroyAllWindows()

face_landmarker.close()
hand_landmarker.close()

print("Video saved as mimi_output.avi")
