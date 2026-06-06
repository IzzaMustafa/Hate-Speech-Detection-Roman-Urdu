import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math

# Initialize
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Screen setup
screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = True  # Move mouse to top-left to abort

# Calibration variables
calibrated = False
calib_left_ratio = None
calib_right_ratio = None
calib_center_ratio = None

def get_eye_landmarks(landmarks, w, h):
    """Extract left and right eye landmarks"""
    # Left eye indices (MediaPipe FaceMesh)
    left_eye = [33, 133, 157, 158, 159, 160, 161, 173]
    right_eye = [362, 263, 387, 386, 385, 384, 398, 466]
    
    left_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in left_eye])
    right_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in right_eye])
    
    return left_pts, right_pts

def get_iris_position(eye_pts):
    """Estimate iris position based on eye corners"""
    # Eye corners
    left_corner = eye_pts[0]  # Left corner of eye
    right_corner = eye_pts[1]  # Right corner of eye
    
    # Eye width
    eye_width = np.linalg.norm(right_corner - left_corner)
    
    # Center of eye bounding box
    eye_center = np.mean(eye_pts, axis=0)
    
    # Ratio of how far iris is from left corner
    # This approximates horizontal gaze
    vec_from_left = eye_center - left_corner
    ratio = vec_from_left[0] / eye_width
    
    return ratio

def calibrate_gaze(face_landmarks, frame_w, frame_h):
    """Simple 3-point calibration: look left, center, right"""
    global calibrated, calib_left_ratio, calib_right_ratio, calib_center_ratio
    
    left_eye, right_eye = get_eye_landmarks(face_landmarks, frame_w, frame_h)
    left_ratio = get_iris_position(left_eye)
    right_ratio = get_iris_position(right_eye)
    avg_ratio = (left_ratio + right_ratio) / 2
    
    return avg_ratio

def ratio_to_screen_x(ratio, calib_left, calib_center, calib_right):
    """Convert gaze ratio to screen X coordinate"""
    if ratio <= calib_center:
        # Left side: map calib_left to 0, calib_center to screen_w/2
        t = (ratio - calib_left) / (calib_center - calib_left)
        return int(t * (screen_w / 2))
    else:
        # Right side: map calib_center to screen_w/2, calib_right to screen_w
        t = (ratio - calib_center) / (calib_right - calib_center)
        return int(screen_w/2 + t * (screen_w/2))

# Main loop
cap = cv2.VideoCapture(0)
print("=== Eye-Controlled Cursor ===")
print("Press 'c' to calibrate (look CENTER then press)")
print("After calibration, look: LEFT, CENTER, RIGHT (press 'c' each time)")
print("Press 'q' to quit")
print("Move mouse to TOP-LEFT corner to emergency stop")

calibration_step = 0
calibration_values = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_eye, right_eye = get_eye_landmarks(landmarks, frame_w, frame_h)
        
        # Draw eyes
        for pt in left_eye.astype(int):
            cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)
        for pt in right_eye.astype(int):
            cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)
        
        # Get gaze ratio
        left_ratio = get_iris_position(left_eye)
        right_ratio = get_iris_position(right_eye)
        avg_ratio = (left_ratio + right_ratio) / 2
        
        # Calibration process
        if not calibrated:
            cv2.putText(frame, f"CALIBRATION STEP {calibration_step + 1}/3", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if calibration_step == 0:
                cv2.putText(frame, "Look LEFT then press 'c'", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            elif calibration_step == 1:
                cv2.putText(frame, "Look CENTER then press 'c'", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Look RIGHT then press 'c'", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        else:
            # Move cursor
            cursor_x = ratio_to_screen_x(avg_ratio, calib_left_ratio, 
                                        calib_center_ratio, calib_right_ratio)
            
            # Smooth cursor (simple moving average)
            if not hasattr(ratio_to_screen_x, "prev_x"):
                ratio_to_screen_x.prev_x = cursor_x
            ratio_to_screen_x.prev_x = int(ratio_to_screen_x.prev_x * 0.7 + cursor_x * 0.3)
            
            pyautogui.moveTo(ratio_to_screen_x.prev_x, screen_h // 2)
            
            cv2.putText(frame, f"Cursor X: {ratio_to_screen_x.prev_x}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Blink detection for click
            # Check eye aspect ratio (simplified)
            left_height = np.linalg.norm(left_eye[2] - left_eye[5])
            left_width = np.linalg.norm(left_eye[0] - left_eye[1])
            if left_height / left_width < 0.2:  # Blink detected
                pyautogui.click()
                cv2.putText(frame, "CLICK!", (300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # Show current gaze ratio
        cv2.putText(frame, f"Gaze Ratio: {avg_ratio:.2f}", (50, frame_h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Eye Control", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    elif key == ord('c') and not calibrated:
        if results.multi_face_landmarks:
            ratio = calibrate_gaze(landmarks, frame_w, frame_h)
            calibration_values.append(ratio)
            calibration_step += 1
            
            if calibration_step == 3:
                calib_left_ratio = calibration_values[0]
                calib_center_ratio = calibration_values[1]
                calib_right_ratio = calibration_values[2]
                calibrated = True
                print("Calibration complete!")
                print(f"Left: {calib_left_ratio:.3f}, Center: {calib_center_ratio:.3f}, Right: {calib_right_ratio:.3f}")

cap.release()
cv2.destroyAllWindows()
