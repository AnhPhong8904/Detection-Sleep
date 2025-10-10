# Cấu hình cho hệ thống phát hiện ngủ gật

# EAR (Eye Aspect Ratio) threshold for drowsiness detection
EAR_THRESHOLD = 0.25

# MAR (Mouth Aspect Ratio) threshold for yawning detection
MAR_THRESHOLD = 0.7

# Số frame liên tiếp để xác nhận trạng thái
CONSEC_FRAMES = 50

# MediaPipe Face Landmarker configuration
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_PRESENCE_CONFIDENCE = 0.2
MIN_TRACKING_CONFIDENCE = 0.5,
NUM_FACES = 1

# Video display settings
WINDOW_NAME = "Face Landmarks"

# Head pose thresholds (degrees)
# Pitch: positive when head down
HEAD_PITCH_DOWN_DEG = 10
# Yaw: magnitude when turning left/right
HEAD_YAW_TURN_DEG = 10
# Roll: magnitude when tilting
HEAD_ROLL_TILT_DEG = 8

# --- ROI configuration ---
# USE_ROI: bật/tắt crop ROI trước khi detect
USE_ROI = True
# ROI_PIXELS: định nghĩa theo (x, y, w, h) theo pixels trên khung full frame.
# Bạn đã cung cấp '500, 300, 780, 600' — nếu đó là x1,y1,x2,y2, chuyển về w,h:
# w = 780 - 500 = 280, h = 600 - 300 = 300
ROI_PIXELS = (500, 300, 280, 300)