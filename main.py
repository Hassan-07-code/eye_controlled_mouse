"""
==============================================================
EYE CONTROLLED MOUSE (Enhanced Version)
==============================================================
Author: Hassan
Description:
    A computer vision–based human-computer interaction system
    that moves the mouse cursor based on right-eye iris motion
    and performs clicks when the user blinks the left eye.

    Uses the latest MediaPipe (Tasks API) Face Landmarker model.
    Includes calibration, smoothing, adaptive blink detection,
    and optimized performance for normal hardware.
==============================================================
"""

# ==============================
# Imports
# ==============================
import cv2
import mediapipe as mp
import pyautogui
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================
# Model Setup (MediaPipe Face Landmarker)
# ==============================
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,     # Better temporal consistency
    num_faces=1
)

detector = FaceLandmarker.create_from_options(options)

# ==============================
# Webcam and Screen Setup
# ==============================
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

screen_w, screen_h = pyautogui.size()

# ==============================
# Cursor Control Parameters
# ==============================
prev_x, prev_y = 0, 0
smooth_factor = 0.35          # 0–1 → higher = faster cursor reaction
CALIB_MIN_X, CALIB_MAX_X = 0.40, 0.60  # Adjust for personal calibration
CALIB_MIN_Y, CALIB_MAX_Y = 0.35, 0.55

# ==============================
# Blink Detection Parameters
# ==============================
last_click_time = 0
CLICK_COOLDOWN = 1.0          # seconds between valid clicks
BLINK_FRAMES_REQUIRED = 3     # blink must persist this many frames
closed_frames = 0
open_ratio_reference = None   # adaptive blink threshold reference

# ==============================
# Processing Loop
# ==============================
while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Process the frame through MediaPipe
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))
    frame_h, frame_w, _ = frame.shape

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        # ====================================================
        # 1. Cursor Control using Right Iris (Landmarks 474–477)
        # ====================================================
        # Compute iris center as average of 4 points
        iris_x = sum(lm.x for lm in landmarks[474:478]) / 4
        iris_y = sum(lm.y for lm in landmarks[474:478]) / 4

        # Visualize the iris points
        for lm in landmarks[474:478]:
            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # green dots

        # Map normalized coordinates to screen with calibration
        norm_x = (iris_x - CALIB_MIN_X) / (CALIB_MAX_X - CALIB_MIN_X)
        norm_y = (iris_y - CALIB_MIN_Y) / (CALIB_MAX_Y - CALIB_MIN_Y)
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))

        screen_x = screen_w * norm_x
        screen_y = screen_h * norm_y

        # Smooth the cursor motion
        prev_x = (1 - smooth_factor) * prev_x + smooth_factor * screen_x
        prev_y = (1 - smooth_factor) * prev_y + smooth_factor * screen_y
        pyautogui.moveTo(prev_x, prev_y, duration=0)

        # ====================================================
        # 2. Blink Detection using Left Eye (Landmarks 145, 159)
        # ====================================================
        left_eye_top = landmarks[145]
        left_eye_bottom = landmarks[159]

        # Draw yellow dots for left-eye landmarks
        for lm in [left_eye_top, left_eye_bottom]:
            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # yellow dots

        blink_ratio = left_eye_top.y - left_eye_bottom.y

        # Initialize reference ratio if not yet set
        if open_ratio_reference is None:
            open_ratio_reference = blink_ratio

        adaptive_threshold = open_ratio_reference * 0.45  # 45% of open-eye distance

        # Detect continuous closure
        if blink_ratio < adaptive_threshold:
            closed_frames += 1
        else:
            closed_frames = 0

        # Trigger click if blink sustained and cooldown elapsed
        if closed_frames > BLINK_FRAMES_REQUIRED:
            current_time = time.time()
            if current_time - last_click_time > CLICK_COOLDOWN:
                pyautogui.click()
                last_click_time = current_time
                closed_frames = 0  # reset after click

    # ====================================================
    # 3. Display Output Frame
    # ====================================================
    cv2.imshow("Eye Controlled Mouse (Enhanced)", frame)

    # Exit on 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# ==============================
# Cleanup
# ==============================
cam.release()
cv2.destroyAllWindows()