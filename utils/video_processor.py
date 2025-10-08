import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks.python import vision
from detect.eye_ratio import compute_eyes_ear, LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from detect.mouth_ratio import compute_mouth_mar, MOUTH_LANDMARKS
from detect.head import compute_head_angles
from utils.draw_alert import draw_bbox_with_label
from utils.config import EAR_THRESHOLD, MAR_THRESHOLD, CONSEC_FRAMES, WINDOW_NAME
from utils.tracker import DetectionTracker

class VideoProcessor:
    """Class xử lý video detection"""
    
    def __init__(self, detector, video_writer=None):
        self.detector = detector
        self.video_writer = video_writer
        self.per_face_low_ear_counters = []
        self.per_face_high_mar_counters = []
        self.tracker = DetectionTracker()
        self.last_blendshapes = None
        
    def start_processing(self):
        """Bắt đầu xử lý video"""
        self.tracker.start_tracking()
        
    def stop_processing(self):
        """Dừng xử lý video"""
        self.tracker.stop_tracking()
        if self.video_writer is not None:
            self.video_writer.release()
            
    def process_frame(self, frame, timestamp):
        """Xử lý một frame"""
        # Increment frame counter
        self.tracker.increment_frame()
        
        # Convert frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect_for_video(mp_image, timestamp)
        
        # Convert back to BGR for display
        output_bgr = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR)
        
        # Draw tracking info
        self._draw_info_panel(output_bgr)
        
        # Process face detection
        if detection_result.face_landmarks:
            self._process_faces(detection_result, output_bgr)
            
        # Lưu blendshapes của frame cuối cùng
        if detection_result.face_blendshapes:
            self.last_blendshapes = detection_result.face_blendshapes[0]
            
        # Write to video if saving
        if self.video_writer is not None:
            self.video_writer.write(output_bgr)
            
        return output_bgr
        
    def _draw_info_panel(self, frame):
        """Vẽ info panel compact"""
        current_time = time.time()
        elapsed_time = current_time - self.tracker.start_time if self.tracker.start_time else 0
        
        # Compact info panel background (smaller)
        cv2.rectangle(frame, (10, 10), (180, 60), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (180, 60), (255, 255, 255), 1)
        
        # Frame and time info (compact)
        cv2.putText(frame, f"F:{self.tracker.frame_count} T:{elapsed_time:.1f}s", (15, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # FPS info
        if elapsed_time > 0:
            current_fps = self.tracker.frame_count / elapsed_time
            cv2.putText(frame, f"FPS:{current_fps:.1f}", (15, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Detection stats (compact)
        drowsy_head_count = sum(1 for event in self.tracker.detection_events if event['type'] == 'DROWSY_HEAD')
        drowsy_yawn_count = sum(1 for event in self.tracker.detection_events if event['type'] == 'DROWSY_YAWN')
        drowsy_count = sum(1 for event in self.tracker.detection_events if event['type'] == 'DROWSY')
        yawning_count = sum(1 for event in self.tracker.detection_events if event['type'] == 'YAWNING')
        
        cv2.putText(frame, f"DH:{drowsy_head_count} DY:{drowsy_yawn_count} D:{drowsy_count} Y:{yawning_count}", (15, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                   
    def _process_faces(self, detection_result, frame):
        """Xử lý phát hiện khuôn mặt"""
        num_faces = len(detection_result.face_landmarks)
        
        # Ensure counters list size matches current faces
        if len(self.per_face_low_ear_counters) < num_faces:
            self.per_face_low_ear_counters.extend([0] * (num_faces - len(self.per_face_low_ear_counters)))
        elif len(self.per_face_low_ear_counters) > num_faces:
            self.per_face_low_ear_counters = self.per_face_low_ear_counters[:num_faces]
            
        if len(self.per_face_high_mar_counters) < num_faces:
            self.per_face_high_mar_counters.extend([0] * (num_faces - len(self.per_face_high_mar_counters)))
        elif len(self.per_face_high_mar_counters) > num_faces:
            self.per_face_high_mar_counters = self.per_face_high_mar_counters[:num_faces]

        height, width = frame.shape[:2]

        for i, face_landmarks in enumerate(detection_result.face_landmarks):
            # Draw landmarks
            self._draw_landmarks(frame, face_landmarks, width, height)
            
            # Calculate EAR, MAR and Head Pose
            _, _, avg_ear = compute_eyes_ear(face_landmarks)
            mar = compute_mouth_mar(face_landmarks)
            head_angles = compute_head_angles(face_landmarks, width, height)
            
            # Update counters
            if avg_ear < EAR_THRESHOLD:
                self.per_face_low_ear_counters[i] += 1
            else:
                self.per_face_low_ear_counters[i] = 0

            if mar > MAR_THRESHOLD:
                self.per_face_high_mar_counters[i] += 1
            else:
                self.per_face_high_mar_counters[i] = 0

            # Determine status
            is_drowsy = self.per_face_low_ear_counters[i] >= CONSEC_FRAMES
            is_yawning = self.per_face_high_mar_counters[i] >= CONSEC_FRAMES

            # Check head pose states
            head_down = head_angles["pitch_state"] == "Head Down"
            head_turned = head_angles["yaw_state"] != "Center"
            head_tilted = head_angles["roll_state"] != "Straight"
            head_moving = head_down or head_turned or head_tilted
            
            # Logic phân loại theo 5 trường hợp
            if is_drowsy and head_moving:
                # TH1: Nhắm mắt + Di chuyển đầu --> Ngủ gật --> Đỏ
                color = (0, 0, 255)  # Red
                head_info = f"P:{head_angles['pitch']:.1f}° Y:{head_angles['yaw']:.1f}° R:{head_angles['roll']:.1f}°"
                label = f"Ngu gat! EAR={avg_ear:.3f} {head_info}"
                status = "DROWSY_HEAD"
            elif is_drowsy and is_yawning:
                # TH2: Mắt nhắm + Ngáp --> Ngáp buồn ngủ --> Vàng
                color = (0, 255, 255)  # Yellow
                label = f"Ngap buon ngu EAR={avg_ear:.3f} MAR={mar:.3f}"
                status = "DROWSY_YAWN"
            elif is_drowsy:
                # TH3: Mắt nhắm liên tục --> Buồn ngủ --> Vàng
                color = (0, 255, 255)  # Yellow
                label = f"Buon ngu EAR={avg_ear:.3f}"
                status = "DROWSY"
            elif is_yawning:
                # TH4: Chỉ ngáp --> Ngáp nhẹ --> Tím
                color = (255, 0, 255)  # Magenta
                label = f"Ngap nhe MAR={mar:.3f}"
                status = "YAWNING"
            else:
                # TH5: Không có gì --> Tỉnh táo --> Xanh
                color = (0, 255, 0)  # Green
                label = f"Tinh tao EAR={avg_ear:.3f} MAR={mar:.3f}"
                status = "AWAKE"

            # Track detection events
            if status in ['DROWSY_HEAD', 'DROWSY_YAWN', 'DROWSY', 'YAWNING']:
                self.tracker.log_detection_event(status, self.tracker.frame_count, avg_ear, mar)

            # Draw bounding box
            self._draw_bounding_box(frame, face_landmarks, width, height, color, label)
            
    def _draw_landmarks(self, frame, face_landmarks, width, height):
        """Vẽ landmarks trên frame"""
        # Draw eye landmarks
        for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
            lm = face_landmarks[idx]
            cx = int(lm.x * width)
            cy = int(lm.y * height)
            cv2.circle(frame, (cx, cy), 1, (255, 255, 0), -1)  # cyan for eyes

        # Draw mouth landmarks
        for idx in MOUTH_LANDMARKS:
            lm = face_landmarks[idx]
            cx = int(lm.x * width)
            cy = int(lm.y * height)
            cv2.circle(frame, (cx, cy), 1, (0, 255, 255), -1)  # yellow for mouth
            
    def _draw_bounding_box(self, frame, face_landmarks, width, height, color, label):
        """Vẽ bounding box và label"""
        # Bounding box from normalized landmarks
        xs = [lm.x for lm in face_landmarks]
        ys = [lm.y for lm in face_landmarks]
        x1 = min(xs) * width
        y1 = min(ys) * height
        x2 = max(xs) * width
        y2 = max(ys) * height

        draw_bbox_with_label(frame, (x1, y1, x2, y2), color=color, label=label)
