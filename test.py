# import cv2

# cap = cv2.VideoCapture(r"video\ngoap1.mp4")
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# duration = frame_count / fps

# print(f"FPS gốc: {fps}")
# print(f"Tổng số frame: {frame_count}")
# print(f"Thời lượng video: {duration:.2f} giây")
# cap.release()
import cv2

# Đọc video
video_path = r"video\ngugat_Trim.mp4"
cap = cv2.VideoCapture(video_path)

# Lấy FPS
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS của video:", fps)

cap.release()
