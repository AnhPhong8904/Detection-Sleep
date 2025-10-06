import numpy as np

MOUTH_LANDMARKS = [61, 291, 13, 14, 78, 82, 308, 312]

def _euclidean_distance(a, b):
    ax, ay = a
    bx, by = b
    return float(np.linalg.norm([ax - bx, ay - by]))

def _extract_points(landmarks, indices):
    return [(landmarks[i].x, landmarks[i].y) for i in indices]

def calculate_mouth_ratio(landmarks, indices):
    """Tính tỷ lệ mở miệng (Mouth Aspect Ratio - MAR)"""
    p1, p2, p3, p4, p5, p6, p7, p8 = _extract_points(landmarks, indices)
    
    # Tính chiều rộng miệng (horizontal distance)
    mouth_width = _euclidean_distance(p2, p8)
    
    # Tính chiều cao miệng (vertical distances)
    mouth_height_1 = _euclidean_distance(p3, p7)
    mouth_height_2 = _euclidean_distance(p4, p6)
    
    # Lấy trung bình chiều cao
    mouth_height = (mouth_height_1 + mouth_height_2) / 3.0
    
    # Tính MAR (Mouth Aspect Ratio)
    if mouth_width == 0:
        return 0.0
    
    mar = mouth_height / mouth_width
    return mar

def compute_mouth_mar(face_landmarks):
    """Tính MAR cho một khuôn mặt"""
    if not face_landmarks:
        return 0.0
    
    mar = calculate_mouth_ratio(face_landmarks, MOUTH_LANDMARKS)
    return mar