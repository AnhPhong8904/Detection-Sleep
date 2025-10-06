# Facial Landmarks Detection - Hệ thống phát hiện ngủ gật và ngáp

Hệ thống phát hiện ngủ gật và ngáp sử dụng MediaPipe Face Landmarks với các thuật toán EAR (Eye Aspect Ratio) và MAR (Mouth Aspect Ratio).

## 🚀 Tính năng

- **Phát hiện ngủ gật**: Sử dụng EAR để phát hiện mắt nhắm
- **Phát hiện ngáp**: Sử dụng MAR để phát hiện miệng mở rộng
- **Real-time tracking**: Theo dõi thời gian, frame counter, FPS
- **Visualization**: Hiển thị landmarks, bounding box với màu sắc
- **Video recording**: Lưu video kết quả detection
- **Compact UI**: Info panel nhỏ gọn không che khuất vật thể

## 📁 Cấu trúc thư mục

```
Facial Landmarks Detection/
├── main.py                          # File chính chạy chương trình
├── face_landmarker.task             # Model MediaPipe
├── detect/
│   ├── eye_ratio.py                # Tính toán EAR (Eye Aspect Ratio)
│   ├── mouth_ratio.py              # Tính toán MAR (Mouth Aspect Ratio)
│   └── facial_landmark.py          # Vẽ landmarks và blendshapes
├── utils/
│   ├── config.py                   # Cấu hình thresholds và settings
│   ├── tracker.py                  # Theo dõi thời gian và events
│   ├── video_processor.py          # Xử lý video và detection
│   └── draw_alert.py               # Vẽ bounding box và alerts
└── video/
    ├── ngugat1.mp4                 # Video mẫu ngủ gật
    ├── ngoap1.mp4                  # Video mẫu ngáp
    └── ...                         # Các video khác
```

## 🛠️ Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy

### Cài đặt dependencies

```bash
pip install opencv-python mediapipe numpy
```

### Tải model MediaPipe

Tải file `face_landmarker.task` từ [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) và đặt vào thư mục gốc.

## 🎯 Sử dụng

### Cú pháp cơ bản

```bash
python main.py --video <nguồn_video> [options]
```

### Tham số

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--video` | Nguồn video (0 cho webcam, hoặc đường dẫn file) | `0` |
| `--model` | Đường dẫn đến model MediaPipe | `face_landmarker.task` |
| `--output` | Đường dẫn file video đầu ra | `output_detection.mp4` |
| `--save` | Lưu video trong quá trình detect | `False` |

### Ví dụ sử dụng

#### 1. Chạy với webcam
```bash
python main.py --video 0
```

#### 2. Chạy với video file
```bash
python main.py --video "video/ngugat1.mp4"
```

#### 3. Chạy và lưu video kết quả
```bash
python main.py --video "video/ngugat1.mp4" --save --output "result.mp4"
```

#### 4. Sử dụng model tùy chỉnh
```bash
python main.py --video "video/test.mp4" --model "models/custom_landmarker.task"
```

## 🎨 Giao diện

### Info Panel (góc trên bên trái)
```
┌─────────────────────────────┐
│ F:1234 T:45.6s             │  ← Frame count và thời gian
│ FPS: 29.8                  │  ← Tốc độ xử lý
│ D:5 Y:3 A:2                │  ← Số lần phát hiện (Drowsy, Yawning, Alert)
└─────────────────────────────┘
```

**Ý nghĩa các số:**
- **D:5** = Đã phát hiện 5 lần ngủ gật
- **Y:3** = Đã phát hiện 3 lần ngáp  
- **A:2** = Đã phát hiện 2 lần cảnh báo (cả ngủ gật và ngáp cùng lúc)

### Màu sắc detection
- 🟢 **Xanh lá**: Tỉnh táo (AWAKE)
- 🟠 **Cam**: Ngủ gật (DROWSY)
- 🟣 **Tím**: Ngáp (YAWNING)
- 🔴 **Đỏ**: Cảnh báo (ALERT - cả ngủ gật và ngáp)

### Landmarks
- **Chấm xanh dương**: Landmarks mắt
- **Chấm vàng**: Landmarks miệng

## ⚙️ Cấu hình

Chỉnh sửa file `utils/config.py` để thay đổi các ngưỡng:

```python
# EAR (Eye Aspect Ratio) threshold for drowsiness detection
EAR_THRESHOLD = 0.25

# MAR (Mouth Aspect Ratio) threshold for yawning detection
MAR_THRESHOLD = 0.5

# Số frame liên tiếp để xác nhận trạng thái
CONSEC_FRAMES = 20

# MediaPipe Face Landmarker configuration
MIN_FACE_DETECTION_CONFIDENCE = 0.2
NUM_FACES = 1
```

## 🔧 Thuật toán

### EAR (Eye Aspect Ratio)
```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```
- p1, p4: Góc ngoài và trong của mắt
- p2, p5: Điểm trên và dưới mắt
- p3, p6: Điểm giữa trên và dưới

### MAR (Mouth Aspect Ratio)
```
MAR = (mouth_height_1 + mouth_height_2) / (2 * mouth_width)
```
- Tính tỷ lệ chiều cao/chiều rộng miệng

## 📊 Output

### Console Output
```
Bắt đầu theo dõi lúc: 2024-01-15 14:30:25
Frame   30 |   1.0s |    AWAKE | EAR: 0.285 | MAR: 0.245
[Frame 45] DROWSY - EAR: 0.220, MAR: 0.180 - 2024-01-15 14:30:26.123
Kết thúc theo dõi lúc: 2024-01-15 14:30:35
Thời gian xử lý: 10.25 giây
Tổng số frame: 307
FPS trung bình: 29.95
```

### Video Output (nếu save)
- Video chứa tất cả annotations
- Info panel real-time
- Bounding box với màu sắc
- Landmarks dots

## 🚨 Xử lý lỗi

### Lỗi thường gặp

1. **"Import could not be resolved"**
   - Kiểm tra cài đặt dependencies
   - Đảm bảo file `.task` có trong thư mục

2. **"Không phát hiện khuôn mặt"**
   - Điều chỉnh `MIN_FACE_DETECTION_CONFIDENCE` trong config
   - Đảm bảo ánh sáng đủ
   - Kiểm tra chất lượng video

3. **Video không hiển thị**
   - Kiểm tra đường dẫn video
   - Đảm bảo webcam được kết nối (nếu dùng --video 0)

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👨‍💻 Tác giả

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Face Landmark Detection
- [OpenCV](https://opencv.org/) - Computer Vision Library
- [NumPy](https://numpy.org/) - Numerical Computing

---

**Lưu ý**: Dự án này chỉ mang tính chất nghiên cứu và học tập. Không sử dụng cho mục đích thương mại mà không có sự cho phép.
