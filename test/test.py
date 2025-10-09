# # import cv2

# # cap = cv2.VideoCapture(r"video\ngoap1.mp4")
# # fps = cap.get(cv2.CAP_PROP_FPS)
# # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# # duration = frame_count / fps

# # print(f"FPS gốc: {fps}")
# # print(f"Tổng số frame: {frame_count}")
# # print(f"Thời lượng video: {duration:.2f} giây")
# # cap.release()
# import cv2

# # Đọc video
# video_path = r"video\ngugat_Trim.mp4"
# cap = cv2.VideoCapture(video_path)

# # Lấy FPS
# fps = cap.get(cv2.CAP_PROP_FPS)
# print("FPS của video:", fps)

# cap.release()
import pandas as pd

# ===== CẤU HÌNH =====
csv_ngu = "ngu_gat.csv"
csv_tinh = "tinh_tao.csv"
output_csv = "ground_truth.csv"
# =====================

# Đọc và gộp
df_ngu = pd.read_csv(csv_ngu)
df_tinh = pd.read_csv(csv_tinh)

df_all = pd.concat([df_ngu, df_tinh], ignore_index=True)

# Ghi ra file hợp nhất
df_all.to_csv(output_csv, index=False, encoding="utf-8")

print(f"✅ Đã gộp thành công thành file {output_csv} (Tổng {len(df_all)} video)")

