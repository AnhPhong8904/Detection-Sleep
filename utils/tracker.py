import time
from datetime import datetime

class DetectionTracker:
    """Class để theo dõi thời gian và frame trong quá trình detection"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.frame_count = 0
        self.detection_events = []
        
    def start_tracking(self):
        """Bắt đầu theo dõi"""
        self.start_time = time.time()
        self.frame_count = 0
        self.detection_events = []
        print(f"Bắt đầu theo dõi lúc: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        
    def stop_tracking(self):
        """Dừng theo dõi"""
        self.end_time = time.time()
        if self.start_time:
            duration = self.end_time - self.start_time
            print(f"Kết thúc theo dõi lúc: {datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Thời gian xử lý: {duration:.2f} giây")
            print(f"Tổng số frame: {self.frame_count}")
            if duration > 0:
                fps = self.frame_count / duration
                print(f"FPS trung bình: {fps:.2f}")
                
    def increment_frame(self):
        """Tăng counter frame"""
        self.frame_count += 1
        
    def log_detection_event(self, event_type, frame_num, ear_value=None, mar_value=None):
        """Ghi lại sự kiện detection"""
        current_time = time.time()
        timestamp = current_time - self.start_time if self.start_time else 0
        
        event = {
            'timestamp': timestamp,
            'frame': frame_num,
            'type': event_type,
            'ear': ear_value,
            'mar': mar_value,
            'datetime': datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        }
        
        self.detection_events.append(event)
        
        # In ra console cho các sự kiện quan trọng
        if event_type in ['DROWSY', 'YAWNING', 'ALERT']:
            print(f"[Frame {frame_num}] {event_type} - EAR: {ear_value:.3f}, MAR: {mar_value:.3f} - {event['datetime']}")
            
    def get_statistics(self):
        """Lấy thống kê tổng quan"""
        if not self.detection_events:
            return {}
            
        drowsy_count = sum(1 for event in self.detection_events if event['type'] == 'DROWSY')
        yawning_count = sum(1 for event in self.detection_events if event['type'] == 'YAWNING')
        alert_count = sum(1 for event in self.detection_events if event['type'] == 'ALERT')
        
        return {
            'total_frames': self.frame_count,
            'total_events': len(self.detection_events),
            'drowsy_events': drowsy_count,
            'yawning_events': yawning_count,
            'alert_events': alert_count,
            'duration': self.end_time - self.start_time if self.end_time and self.start_time else 0
        }
        
    def save_log_to_file(self, filename="detection_log.txt"):
        """Lưu log ra file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=== DETECTION LOG ===\n")
                f.write(f"Thời gian bắt đầu: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Thời gian kết thúc: {datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Tổng số frame: {self.frame_count}\n")
                f.write(f"Thời gian xử lý: {self.end_time - self.start_time:.2f} giây\n\n")
                
                f.write("=== EVENTS ===\n")
                for event in self.detection_events:
                    f.write(f"[{event['datetime']}] Frame {event['frame']:4d} | "
                           f"{event['type']:8s} | EAR: {event['ear']:.3f} | MAR: {event['mar']:.3f}\n")
                           
                # Thống kê
                stats = self.get_statistics()
                f.write("\n=== STATISTICS ===\n")
                f.write(f"Tổng số sự kiện: {stats['total_events']}\n")
                f.write(f"Sự kiện ngủ gật: {stats['drowsy_events']}\n")
                f.write(f"Sự kiện ngáp: {stats['yawning_events']}\n")
                f.write(f"Sự kiện cảnh báo: {stats['alert_events']}\n")
                
            print(f"Log đã được lưu vào file: {filename}")
            
        except Exception as e:
            print(f"Lỗi khi lưu log: {e}")
            
    def print_frame_info(self, frame_num, ear_value, mar_value, status):
        """In thông tin frame hiện tại"""
        if frame_num % 30 == 0:  # In mỗi 30 frame
            current_time = time.time()
            elapsed = current_time - self.start_time if self.start_time else 0
            print(f"Frame {frame_num:4d} | {elapsed:6.1f}s | {status:8s} | EAR: {ear_value:.3f} | MAR: {mar_value:.3f}")
