"""
Multi-Object Tracking System for Highway Driving Videos
Uses YOLOv8 for detection and BoT-SORT for tracking
"""


import cv2
import numpy as np
from pathlib import Path
import csv
from collections import defaultdict, deque
from ultralytics import YOLO
import torch


class VehicleTracker:
  
   def __init__(self, model_name='yolov8x.pt', conf_threshold=0.3, iou_threshold=0.5):
       self.conf_threshold = conf_threshold
       self.iou_threshold = iou_threshold
       self.model = YOLO(model_name)
      
       # COCO class IDs for relevant objects
       self.target_classes = {
           0: 'Person',      # Pedestrian
           1: 'Bicycle',
           2: 'Car',
           3: 'Motorcycle',
           5: 'Bus',
           7: 'Truck'
       }
      
       # Map to simplified classes
       self.class_mapping = {
           'Person': 'Pedestrian',
           'Bicycle': 'Motorcycle',
           'Car': 'Car',
           'Motorcycle': 'Motorcycle',
           'Bus': 'Bus',
           'Truck': 'Truck'
       }
      
       # Colors for different classes (BGR format)
       self.colors = {
           'Car': (255, 0, 0),
           'Truck': (0, 255, 0),
           'Motorcycle': (0, 0, 255),
           'Bus': (255, 255, 0),
           'Pedestrian': (255, 0, 255)
       }
      
       # Track history for velocity calculation
       self.track_history = defaultdict(lambda: deque(maxlen=30))
       self.track_velocities = {}
      
       # Statistics
       self.frame_count = 0
       self.total_tracks = 0
      
   def calculate_velocity(self, track_id, bbox, frame_number, fps=30):
       center_x = (bbox[0] + bbox[2]) / 2
       center_y = (bbox[1] + bbox[3]) / 2
      
       self.track_history[track_id].append({
           'frame': frame_number,
           'center': (center_x, center_y)
       })
      
       if len(self.track_history[track_id]) < 2:
           return 0.0, 0.0
      
       # Calculate velocity using last N frames
       history = list(self.track_history[track_id])
       if len(history) >= 10:
           # Use last 10 frames for smoother velocity estimation
           old_pos = history[-10]
           new_pos = history[-1]
          
           frame_diff = new_pos['frame'] - old_pos['frame']
           if frame_diff > 0:
               dx = new_pos['center'][0] - old_pos['center'][0]
               dy = new_pos['center'][1] - old_pos['center'][1]
              
               # Convert to velocity (pixels per second)
               vx = (dx / frame_diff) * fps
               vy = (dy / frame_diff) * fps
              
               # Store velocity
               self.track_velocities[track_id] = (vx, vy)
               return vx, vy
      
       return self.track_velocities.get(track_id, (0.0, 0.0))
  
   def draw_tracking_info(self, frame, bbox, track_id, class_name, confidence, velocity):
       x1, y1, x2, y2 = map(int, bbox)
       color = self.colors.get(class_name, (255, 255, 255))
      
       # Draw bounding box
       cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
      
       # Calculate velocity magnitude
       speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
      
       # Prepare label with ID, class, and velocity
       label = f"ID:{track_id} {class_name}"
       velocity_label = f"Speed:{speed:.1f}px/s"
      
       # Draw background for labels
       label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
       velocity_size, _ = cv2.getTextSize(velocity_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
      
       # Label background
       cv2.rectangle(frame, (x1, y1 - label_size[1] - 35),
                    (x1 + max(label_size[0], velocity_size[0]) + 10, y1),
                    color, -1)
      
       # Draw text
       cv2.putText(frame, label, (x1 + 5, y1 - 20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
       cv2.putText(frame, velocity_label, (x1 + 5, y1 - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
      
       # Draw velocity vector (optional visualization)
       if speed > 5:  # Only draw if speed is significant
           center_x = int((x1 + x2) / 2)
           center_y = int((y1 + y2) / 2)
           end_x = int(center_x + velocity[0] / 10)
           end_y = int(center_y + velocity[1] / 10)
           cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                         (0, 255, 255), 2, tipLength=0.3)
  
   def process_video(self, input_path, output_video_path, output_csv_path):
       # Open video
       cap = cv2.VideoCapture(input_path)
       if not cap.isOpened():
           raise ValueError(f"Could not open video: {input_path}")
      
       # Get video properties
       fps = int(cap.get(cv2.CAP_PROP_FPS))
       width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
       height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

       codecs = ['avc1', 'H264', 'X264', 'mp4v']
       out = None
      
       for codec in codecs:
           try:
               fourcc = cv2.VideoWriter_fourcc(*codec)
               test_out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
               if test_out.isOpened():
                   out = test_out
                   break
               test_out.release()
           except:
               continue
      
       if out is None:
           raise RuntimeError("Codec not present")
      
       # Initialize CSV writer
       csv_file = open(output_csv_path, 'w', newline='')
       csv_writer = csv.writer(csv_file)
       csv_writer.writerow(['frame_id', 'object_id', 'class', 'bbox_x1', 'bbox_y1',
                           'bbox_x2', 'bbox_y2', 'confidence', 'velocity_x', 'velocity_y'])
      
       frame_number = 0
      
       try:
           while cap.isOpened():
               ret, frame = cap.read()
               if not ret:
                   break
              
               frame_number += 1
              
               # Run YOLOv8 tracking (with built-in BoT-SORT)
               results = self.model.track(
                   frame,
                   persist=True,
                   conf=self.conf_threshold,
                   iou=self.iou_threshold,
                   classes=list(self.target_classes.keys()),
                   tracker="botsort.yaml",
                   verbose=False
               )
              
               # Process detections
               if results[0].boxes is not None and results[0].boxes.id is not None:
                   boxes = results[0].boxes.xyxy.cpu().numpy()
                   track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                   confidences = results[0].boxes.conf.cpu().numpy()
                   classes = results[0].boxes.cls.cpu().numpy().astype(int)
                  
                   for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                       # Get class name
                       class_name = self.target_classes.get(cls, 'Unknown')
                       mapped_class = self.class_mapping.get(class_name, class_name)
                      
                       # Calculate velocity
                       vx, vy = self.calculate_velocity(track_id, box, frame_number, fps)
                      
                       # Draw tracking info
                       self.draw_tracking_info(frame, box, track_id, mapped_class, conf, (vx, vy))
                      
                       # Write to CSV
                       csv_writer.writerow([
                           frame_number, track_id, mapped_class,
                           int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                           f"{conf:.3f}", f"{vx:.2f}", f"{vy:.2f}"
                       ])
              
               # Add frame info
               info_text = f"Frame: {frame_number}/{total_frames}"
               cv2.putText(frame, info_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
              
               # Write frame
               out.write(frame)
              
               # Progress update
               if frame_number % 30 == 0:
                   progress = (frame_number / total_frames) * 100
                   print(f"Progress: {progress:.1f}% ({frame_number}/{total_frames} frames)")
      
       finally:
           # Cleanup
           cap.release()
           out.release()
           csv_file.close()
           cv2.destroyAllWindows()
      
       print(f"\nProcessing complete!")

def main():
  
   # Configuration
   INPUT_VIDEO = "challenge-mle2.mp4"
   OUTPUT_VIDEO = "output_tracking.mp4"
   OUTPUT_CSV = "tracking_data.csv"
  
   # Model configuration
   MODEL = "yolov8x.pt"
   CONF_THRESHOLD = 0.3
   IOU_THRESHOLD = 0.5
  
   # Check if input video exists
   if not Path(INPUT_VIDEO).exists():
       print(f"Error: Input video '{INPUT_VIDEO}' not found!")
       print("Please place the video file in the current directory.")
       return
  
   # Initialize tracker
   tracker = VehicleTracker(
       model_name=MODEL,
       conf_threshold=CONF_THRESHOLD,
       iou_threshold=IOU_THRESHOLD
   )
  
   # Process video
   tracker.process_video(INPUT_VIDEO, OUTPUT_VIDEO, OUTPUT_CSV)
   print(f"  - Video: {OUTPUT_VIDEO}")
   print(f"  - CSV: {OUTPUT_CSV}")




if __name__ == "__main__":
   main()
