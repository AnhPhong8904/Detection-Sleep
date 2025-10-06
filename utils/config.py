# Cấu hình cho hệ thống phát hiện ngủ gật

# EAR (Eye Aspect Ratio) threshold for drowsiness detection
EAR_THRESHOLD = 0.25

# MAR (Mouth Aspect Ratio) threshold for yawning detection
MAR_THRESHOLD = 0.5

# Số frame liên tiếp để xác nhận trạng thái
CONSEC_FRAMES = 20

# MediaPipe Face Landmarker configuration
MIN_FACE_DETECTION_CONFIDENCE = 0.2
NUM_FACES = 1

# Video display settings
WINDOW_NAME = "Face Landmarks"
