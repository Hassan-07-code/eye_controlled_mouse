"""
Eye Controlled Mouse — improved version
- Calibration routine (press 'c' to run calibration)
- Iris-based cursor mapping with per-user calibration
- Exponential smoothing + small median filter to remove spikes
- Robust blink detection using normalized vertical/horizontal ratio (EAR-like)
- Safe clamping to avoid corners (preserves PyAutoGUI failsafe if enabled)
"""

import cv2
import mediapipe as mp
import pyautogui
import time
import collections
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# Model setup
# -----------------------------
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=r"E:\BS AI\6th Semester\Computer Vision\Projects\eye_controlled_mouse\face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)
detector = FaceLandmarker.create_from_options(options)

# -----------------------------
# Webcam and screen
# -----------------------------
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

screen_w, screen_h = pyautogui.size()

# -----------------------------
# Parameters (tune these)
# -----------------------------
# Cursor smoothing
prev_x, prev_y = screen_w/2, screen_h/2
smooth_factor = 0.30           # 0..1 (higher = faster)
median_buffer_size = 5         # median filter length (odd)
median_buf_x = collections.deque(maxlen=median_buffer_size)
median_buf_y = collections.deque(maxlen=median_buffer_size)

# Calibration (default fallbacks)
calib_min_x, calib_max_x = 0.40, 0.60
calib_min_y, calib_max_y = 0.35, 0.55

# Blink detection
last_click_time = 0.0
CLICK_COOLDOWN = 0.35          # seconds
closed_frames = 0
BLINK_FRAMES_REQUIRED = 1      # frames of sustained "closed" to accept as blink
OPEN_REF_ALPHA = 0.05          # smoothing for open-eye reference
adaptive_open_ref = None

# Safe margins to avoid screen corners
SAFE_MARGIN = 10  # pixels

# Toggle PyAutoGUI failsafe (recommended True but we clamp)
pyautogui.FAILSAFE = True

# Utility: median of deque
def median_of_deque(d):
    if not d:
        return None
    s = sorted(d)
    return s[len(s)//2]

# -----------------------------
# Calibration routine
#    Press 'c' in the GUI window to run calibration:
#    1) Look center for 2s -> records center
#    2) Look left, right, top, bottom for 2s each
#    It will compute min/max ranges for mapping.
# -----------------------------
def run_calibration(frames=40, wait_between=0.5):
    global calib_min_x, calib_max_x, calib_min_y, calib_max_y
    print("Calibration started. Follow prompts and look at indicated point.")
    samples = {'center':[], 'left':[], 'right':[], 'top':[], 'bottom':[]}
    prompts = [('center','Look at CENTER'),
               ('left','Look at LEFT edge'),
               ('right','Look at RIGHT edge'),
               ('top','Look at TOP edge'),
               ('bottom','Look at BOTTOM edge')]
    for key, msg in prompts:
        print(msg)
        time.sleep(wait_between)
        collected = []
        for _ in range(frames):
            ret, frame = cam.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect_for_video(mp_img, int(time.time()*1000))
            if res.face_landmarks:
                lm = res.face_landmarks[0]
                iris_x = sum(p.x for p in lm[474:478]) / 4
                iris_y = sum(p.y for p in lm[474:478]) / 4
                collected.append((iris_x, iris_y))
            # small delay
            cv2.waitKey(1)
        if collected:
            xs = [c[0] for c in collected]
            ys = [c[1] for c in collected]
            samples[key] = (sum(xs)/len(xs), sum(ys)/len(ys))
        else:
            samples[key] = None
    # compute calib ranges from left/right/top/bottom sample points
    if samples['left'] and samples['right']:
        calib_min_x = min(samples['left'][0], samples['center'][0]) - 0.02
        calib_max_x = max(samples['right'][0], samples['center'][0]) + 0.02
    if samples['top'] and samples['bottom']:
        calib_min_y = min(samples['top'][1], samples['center'][1]) - 0.02
        calib_max_y = max(samples['bottom'][1], samples['center'][1]) + 0.02
    # clamp sensible
    calib_min_x = max(0.0, calib_min_x)
    calib_max_x = min(1.0, calib_max_x)
    calib_min_y = max(0.0, calib_min_y)
    calib_max_y = min(1.0, calib_max_y)
    print("Calibration finished.")
    print(f"calib_min_x={calib_min_x:.3f}, calib_max_x={calib_max_x:.3f}")
    print(f"calib_min_y={calib_min_y:.3f}, calib_max_y={calib_max_y:.3f}")

# -----------------------------
# Main loop
# -----------------------------
last_frame_time = time.time()
while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_im = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # detect
    result = detector.detect_for_video(mp_im, int(time.time()*1000))

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        # --------- iris center (right eye) ----------
        iris_x = sum(p.x for p in lm[474:478]) / 4
        iris_y = sum(p.y for p in lm[474:478]) / 4

        # visualize
        for p in lm[474:478]:
            px = int(p.x * frame_w)
            py = int(p.y * frame_h)
            cv2.circle(frame, (px, py), 2, (0,255,0), -1)

        # map via calibrated ranges
        norm_x = (iris_x - calib_min_x) / (calib_max_x - calib_min_x) if calib_max_x - calib_min_x != 0 else 0.5
        norm_y = (iris_y - calib_min_y) / (calib_max_y - calib_min_y) if calib_max_y - calib_min_y != 0 else 0.5
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        target_x = norm_x * screen_w
        target_y = norm_y * screen_h

        # median buffer + EMA smoothing
        median_buf_x.append(target_x)
        median_buf_y.append(target_y)
        med_x = median_of_deque(median_buf_x)
        med_y = median_of_deque(median_buf_y)
        if med_x is None:
            med_x, med_y = target_x, target_y

        prev_x = (1 - smooth_factor) * prev_x + smooth_factor * med_x
        prev_y = (1 - smooth_factor) * prev_y + smooth_factor * med_y

        # clamp to safe screen area (avoid PyAutoGUI corner triggers)
        safe_x = max(SAFE_MARGIN, min(screen_w - SAFE_MARGIN, prev_x))
        safe_y = max(SAFE_MARGIN, min(screen_h - SAFE_MARGIN, prev_y))

        pyautogui.moveTo(safe_x, safe_y, duration=0)

        # --------- blink detection (left eye) ----------
        top = lm[145]; bottom = lm[159]
        left_corner = lm[33]; right_corner = lm[133]   # approximated eye corners

        # draw blink points
        for p in (top, bottom, left_corner, right_corner):
            cv2.circle(frame, (int(p.x*frame_w), int(p.y*frame_h)), 2, (0,255,255), -1)

        # compute vertical & horizontal distances (in normalized coords)
        v = abs(top.y - bottom.y)
        h = abs(right_corner.x - left_corner.x) + 1e-6
        ear = v / h   # smaller = more closed

        # initialize adaptive_open_ref as running average of ear when open
        if adaptive_open_ref is None:
            adaptive_open_ref = ear
        else:
            adaptive_open_ref = (1 - OPEN_REF_ALPHA) * adaptive_open_ref + OPEN_REF_ALPHA * ear

        # determine threshold (relative)
        ear_threshold = adaptive_open_ref * 0.6   # 0.5..0.7 range for subtle blink

        # count frames
        if ear < ear_threshold:
            closed_frames += 1
        else:
            closed_frames = max(0, closed_frames - 1)

        # click if sustained closed frames and cooldown passed
        now = time.time()
        if closed_frames >= BLINK_FRAMES_REQUIRED and (now - last_click_time) > CLICK_COOLDOWN:
            pyautogui.click()
            last_click_time = now
            closed_frames = 0

    # show
    cv2.putText(frame, "Press 'c' to calibrate, 'q' to quit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.imshow("Eye Controlled Mouse", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        run_calibration(frames=30)

# cleanup
cam.release()
cv2.destroyAllWindows()