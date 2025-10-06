import cv2
import argparse
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from detect.facial_landmark import plot_face_blendshapes_bar_graph
from utils.config import MIN_FACE_DETECTION_CONFIDENCE, NUM_FACES, WINDOW_NAME
from utils.video_processor import VideoProcessor

def main():
    """Hàm main"""
    parser = argparse.ArgumentParser(description="Hệ thống phát hiện ngủ gật")
    parser.add_argument("--video", type=str, default="0", 
                       help="Nguồn video (0 cho webcam, hoặc đường dẫn file)")
    parser.add_argument("--model", type=str, default=r"face_landmarker.task",
                       help="Đường dẫn đến model Mediapipe")
    parser.add_argument("--output", type=str, default="output_detection.mp4",
                       help="Đường dẫn file video đầu ra")
    parser.add_argument("--save", action="store_true",
                       help="Lưu video trong quá trình detect")
    
    args = parser.parse_args()
    
    # Initialize MediaPipe Face Landmarker
    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
        # min_face_presence_confidence
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=NUM_FACES,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Initialize video capture
    if args.video == "0":
        cap = cv2.VideoCapture(0)  # Webcam
        print("Đang sử dụng webcam...")
    else:
        cap = cv2.VideoCapture(args.video)
        print(f"Đang mở video: {args.video}")
        
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {args.video}")
        return
    else:
        print("Video đã được mở thành công!")
    
    # Get video properties for VideoWriter
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if args.video != "0" else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize VideoWriter if save is enabled
    video_writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Đang lưu video vào: {args.output}")

    # Initialize video processor
    video_processor = VideoProcessor(detector, video_writer)
    video_processor.start_processing()
    
    all_blendshapes = []   # lưu điểm số của frame cuối cùng hoặc trung bình

    print("Bắt đầu xử lý video...")
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Đã đọc hết video hoặc lỗi đọc frame")
            break

        timestamp = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_count += 1
        
        # Process frame through VideoProcessor
        output_bgr = video_processor.process_frame(frame, timestamp)

        cv2.imshow(WINDOW_NAME, output_bgr)
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"Đã xử lý {frame_count} frames...")

        if cv2.waitKey(1) & 0xFF == 27:
            print("Người dùng nhấn ESC để thoát")
            break

    cap.release()
    
    # Stop video processing
    video_processor.stop_processing()
    
    if video_writer is not None:
        print(f"Video đã được lưu thành công: {args.output}")
    
    cv2.destroyAllWindows()
    
    # Vẽ biểu đồ blendshapes sau khi video kết thúc
    print("Đang vẽ biểu đồ blendshapes...")
    try:
        from detect.facial_landmark import plot_face_blendshapes_bar_graph
        
        # Lấy blendshapes từ VideoProcessor
        if hasattr(video_processor, 'last_blendshapes') and video_processor.last_blendshapes:
            plot_face_blendshapes_bar_graph(video_processor.last_blendshapes)
            print("Biểu đồ blendshapes đã được hiển thị!")
        else:
            print("Không có dữ liệu blendshapes để hiển thị")
            print(f"VideoProcessor có last_blendshapes: {hasattr(video_processor, 'last_blendshapes')}")
            if hasattr(video_processor, 'last_blendshapes'):
                print(f"Giá trị last_blendshapes: {video_processor.last_blendshapes}")
    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ blendshapes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()