import numpy as np


# MediaPipe FaceMesh landmark indices for eyes
# Using the common 6-point set per eye for EAR calculation
# Left eye: 33 (p1), 160 (p2), 158 (p3), 133 (p4), 153 (p5), 144 (p6)
# Right eye: 263 (p1), 387 (p2), 385 (p3), 362 (p4), 380 (p5), 373 (p6)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [263, 387, 385, 362, 380, 373]


def _euclidean_distance(a, b):
    ax, ay = a
    bx, by = b
    return float(np.linalg.norm([ax - bx, ay - by]))


def _extract_points(landmarks, indices):
    return [(landmarks[i].x, landmarks[i].y) for i in indices]


def calculate_eye_aspect_ratio(landmarks, indices):
    """Calculate EAR for one eye from normalized landmarks and 6 indices.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1, p2, p3, p4, p5, p6 = _extract_points(landmarks, indices)
    vertical_1 = _euclidean_distance(p2, p6)
    vertical_2 = _euclidean_distance(p3, p5)
    horizontal = _euclidean_distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_eyes_ear(face_landmarks):
    """Return (left_ear, right_ear, average_ear) given a list of 468 landmarks."""
    left_ear = calculate_eye_aspect_ratio(face_landmarks, LEFT_EYE_INDICES)
    right_ear = calculate_eye_aspect_ratio(face_landmarks, RIGHT_EYE_INDICES)
    average_ear = (left_ear + right_ear) / 2.0
    return left_ear, right_ear, average_ear


