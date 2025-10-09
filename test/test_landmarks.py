#!/usr/bin/env python3
"""
Script test để kiểm tra chức năng vẽ landmarks của mắt và môi
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from detect.eye_ratio import LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from detect.mouth_ratio import MOUTH_LANDMARKS
from utils.draw_alert import draw_landmarks_visualization

def test_landmarks_visualization():
    """Test vẽ landmarks trên ảnh"""
    
    # Load model Mediapipe FaceLandmarker
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        min_face_detection_confidence=0.5,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    
    # Đọc ảnh test
    image_path = r"video\business-person.png"
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return
    
    print(f"Đã đọc ảnh: {image_path}")
    print(f"Kích thước ảnh: {img.shape}")
    
    # Convert sang RGB cho MediaPipe
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    
    # Detect landmarks
    detection_result = detector.detect(mp_image)
    
    if not detection_result.face_landmarks:
        print("Không phát hiện khuôn mặt trong ảnh")
        return
    
    print(f"Phát hiện {len(detection_result.face_landmarks)} khuôn mặt")
    
    # Vẽ landmarks cho khuôn mặt đầu tiên
    face_landmarks = detection_result.face_landmarks[0]
    
    # Tạo bản copy để vẽ landmarks
    output_img = img.copy()
    
    # Vẽ landmarks
    output_img = draw_landmarks_visualization(
        output_img, face_landmarks, LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_LANDMARKS
    )
    
    # Hiển thị kết quả
    cv2.imshow("Original Image", img)
    cv2.imshow("Landmarks Visualization", output_img)
    
    print("\nCác điểm landmark được vẽ:")
    print(f"Mắt trái (màu xanh lá): {LEFT_EYE_INDICES}")
    print(f"Mắt phải (màu xanh dương): {RIGHT_EYE_INDICES}")
    print(f"Môi (màu hồng): {MOUTH_LANDMARKS}")
    print("\nNhấn phím bất kỳ để thoát...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_landmarks_visualization()
