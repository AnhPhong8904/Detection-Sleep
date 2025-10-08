import numpy as np
import cv2
from utils.config import HEAD_PITCH_DOWN_DEG, HEAD_YAW_TURN_DEG, HEAD_ROLL_TILT_DEG

# Chỉ số landmark dùng cho solvePnP (Mediapipe 468 points)
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_TEMPLE = 234
RIGHT_TEMPLE = 454

_MODEL_POINTS_3D = np.array([
    [0.0,   0.0,   0.0],    # nose tip
    [0.0, -90.0, -10.0],    # chin
    [-60.0, 40.0, -30.0],   # left eye corner
    [ 60.0, 40.0, -30.0],   # right eye corner
    [-70.0, 10.0, -30.0],   # left temple
    [ 70.0, 10.0, -30.0],   # right temple
], dtype=np.float64)

def _landmarks_to_2d_points(face_landmarks, img_w, img_h):
    idxs = [NOSE_TIP, CHIN, LEFT_EYE_CORNER, RIGHT_EYE_CORNER, LEFT_TEMPLE, RIGHT_TEMPLE]
    pts = []
    for i in idxs:
        lm = face_landmarks[i]
        pts.append([lm.x * img_w, lm.y * img_h])
    return np.array(pts, dtype=np.float64)

def _rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        yaw = np.arctan2(R[2, 0], sy)
        pitch = np.arctan2(-R[2, 1], R[2, 2])
        roll = np.arctan2(-R[1, 0], R[0, 0])
    else:
        yaw = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 1], R[2, 2])
        roll = 0.0
    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

def compute_head_angles(face_landmarks, img_w, img_h):
    """Tính (pitch, yaw, roll) bằng solvePnP và phân loại trạng thái."""
    image_points = _landmarks_to_2d_points(face_landmarks, img_w, img_h)

    focal_length = img_w
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        # Fallback: trả về trạng thái trung lập
        return {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "pitch_state": "Normal",
            "yaw_state": "Center",
            "roll_state": "Straight"
        }

    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = _rotation_matrix_to_euler_angles(R)

    pitch_state = "Head Down" if pitch > HEAD_PITCH_DOWN_DEG else "Normal"
    yaw_state = (
        "Turn Left" if yaw < -HEAD_YAW_TURN_DEG else
        "Turn Right" if yaw > HEAD_YAW_TURN_DEG else
        "Center"
    )
    roll_state = (
        "Tilt Right" if roll > HEAD_ROLL_TILT_DEG else
        "Tilt Left" if roll < -HEAD_ROLL_TILT_DEG else
        "Straight"
    )

    return {
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "pitch_state": pitch_state,
        "yaw_state": yaw_state,
        "roll_state": roll_state
    }
