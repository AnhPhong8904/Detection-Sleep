import numpy as np
import cv2
from utils.config import HEAD_PITCH_DOWN_DEG, HEAD_YAW_TURN_DEG, HEAD_ROLL_TILT_DEG

# Landmarks cho head pose estimation (Mediapipe 468 points)
NOSE_INDICES = [1, 4, 5]
CHIN_INDICES = [152, 200, 199]
LEFT_EYE_INDICES = [33, 133]
RIGHT_EYE_INDICES = [263, 362]
LEFT_TEMPLE_INDICES = [234, 127]
RIGHT_TEMPLE_INDICES = [454, 356]

def _avg_point(landmarks, indices):
    """Tính trung bình (x, y, z) của một nhóm landmarks."""
    pts = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices])
    return np.mean(pts, axis=0)

def compute_head_pose(face_landmarks):
    """Tính pitch, yaw, roll (độ) từ Mediapipe FaceMesh landmarks (3D normalized)."""
    nose = _avg_point(face_landmarks, NOSE_INDICES)
    chin = _avg_point(face_landmarks, CHIN_INDICES)
    left_eye = _avg_point(face_landmarks, LEFT_EYE_INDICES)
    right_eye = _avg_point(face_landmarks, RIGHT_EYE_INDICES)
    left_temple = _avg_point(face_landmarks, LEFT_TEMPLE_INDICES)
    right_temple = _avg_point(face_landmarks, RIGHT_TEMPLE_INDICES)

    # --- Góc tính bằng arctan2 ---
    # Pitch (gật lên/xuống)
    pitch = np.degrees(np.arctan2(chin[2] - nose[2], chin[1] - nose[1]))

    # Yaw (xoay trái/phải)
    yaw = np.degrees(np.arctan2(nose[0] - left_temple[0], nose[2] - left_temple[2]))

    # Roll (nghiêng đầu)
    roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    return pitch, yaw, roll

def compute_head_angles(face_landmarks):
    """Return (pitch, yaw, roll, states) từ danh sách landmarks (468 điểm)."""
    pitch, yaw, roll = compute_head_pose(face_landmarks)

    # Phân loại trạng thái (có thể tinh chỉnh ngưỡng qua config)
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
