import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import time
from collections import deque

# ================= CONFIG =================
SMOOTHING = 5
BLINK_THRESHOLD = 0.20
BLINK_COOLDOWN = 1.0
MODEL = r"E:\\BS AI\\6th Semester\\Computer Vision\\Projects\\eye_controlled_mouse\\models\\face_landmarker.task"
# =========================================



class VideoStream:
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)

        if not self.cap.isOpened():
            raise Exception("Camera not accessible")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()


class FaceTracker:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

    def get_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect(mp_image)

        if result.face_landmarks:
            return result.face_landmarks[0]   # same structure (list of landmarks)

        return None


class EyeTracker:
    def get_iris_position(self, landmarks, w, h):
        iris = landmarks[474:478]
        x = int(iris[1].x * w)
        y = int(iris[1].y * h)
        return x, y

    def eye_aspect_ratio(self, landmarks):
        top = landmarks[159].y
        bottom = landmarks[145].y
        left = landmarks[33].x
        right = landmarks[133].x

        vertical = abs(top - bottom)
        horizontal = abs(left - right)

        return vertical / horizontal if horizontal != 0 else 0


class CalibrationManager:
    def __init__(self):
        self.points = []
        self.screen_points = [
            (0.1, 0.1),
            (0.9, 0.1),
            (0.1, 0.9),
            (0.9, 0.9)
        ]
        self.calibrated = False

    def add_point(self, x, y):
        if len(self.points) < 4:
            self.points.append((x, y))

        if len(self.points) == 4:
            self.calibrated = True

    def map_to_screen(self, x, y, screen_w, screen_h):
        if not self.calibrated:
            return None 

        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        norm_x = (x - min_x) / (max_x - min_x + 1e-6)
        norm_y = (y - min_y) / (max_y - min_y + 1e-6)
    

        return int(norm_x * screen_w), int(norm_y * screen_h)


class CursorController:
    def __init__(self, alpha=0.2):
        self.prev_x = None
        self.prev_y = None
        self.alpha = alpha

    def smooth_move(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y

        curr_x = self.prev_x + self.alpha * (x - self.prev_x)
        curr_y = self.prev_y + self.alpha * (y - self.prev_y)

        pyautogui.moveTo(int(curr_x), int(curr_y))

        self.prev_x, self.prev_y = curr_x, curr_y


class BlinkDetector:
    def __init__(self):
        self.last_click = 0
        self.last_blink_time = 0
        self.ear_history = deque(maxlen=20)

        self.double_click_window = 0.5  # seconds

    def detect(self, ear):
        self.ear_history.append(ear)

        avg_ear = np.mean(self.ear_history)
        threshold = avg_ear * 0.6

        current_time = time.time()

        if ear < threshold:
            # check double blink
            if current_time - self.last_blink_time < self.double_click_window:
                self.last_blink_time = 0
                return "double"

            # single blink
            if current_time - self.last_click > BLINK_COOLDOWN:
                self.last_blink_time = current_time
                self.last_click = current_time
                return "single"

        return None
    
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# ================= MAIN =================

def main():
    stream = VideoStream(0)
    face = FaceTracker()
    eye = EyeTracker()
    calib = CalibrationManager()
    cursor = CursorController()
    blink = BlinkDetector()

    screen_w, screen_h = pyautogui.size()

    while True:
        frame = stream.get_frame()
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)
        frame = preprocess(frame)   # <-- added

        h, w, _ = frame.shape

        landmarks = face.get_landmarks(frame)

        if landmarks:
            x, y = eye.get_iris_position(landmarks, w, h)

            # Calibration (press 'c')
            key = cv2.waitKey(1)

            if key == ord('c'):
                calib.add_point(x, y)
                print(f"Calibration point {len(calib.points)}/4")

            screen_coords = calib.map_to_screen(x, y, screen_w, screen_h)

            if screen_coords:
                cursor.smooth_move(*screen_coords)

            ear = eye.eye_aspect_ratio(landmarks)

            action = blink.detect(ear)
            if action == "single":
                pyautogui.click()

            elif action == "double":
                pyautogui.doubleClick()

        cv2.imshow("Eye Mouse", cv2.resize(frame, (800, 600)))

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            break
    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()