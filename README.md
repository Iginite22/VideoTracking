# VideoTracking:- Multi-Class Object Detection and Tracking in Highway Driving

## Models Used for Detection

**YOLOv8x**

- Confidence threshold: 0.3, IoU threshold: 0.5
- Target classes: Car, Truck, Bus, Motorcycle, Pedestrian

## Tracking Methodology

**BoT-SORT Algorithm**

- This algorithm uses Kalman filter which helps in predicting object position between frames.
- We considered High-confidence (>50%) then low-confidence (30-50%) matching.
- We have used 30-frame persistence buffer, sequential ID assignment

**Velocity Calculation:** Temporal differencing over 10-frame window (pixels/second)

## Performance Optimization Steps

1. **YOLOv8x:** Pre-trained weights, no training required
2. **Detection Filtering:** Confidence threshold + class filtering + Non-Maximum Suppression
3. **Data Structures:** NumPy arrays, deque (O(1)), dictionaries (O(1))
4. **Streaming Processing:** Frame-by-frame, low memory

## Handling Occlusion, Entry/Exit, and Frame Drops

**Occlusion:**

- Kalman filter predicts position during occlusion
- 30-frame track persistence
- ReID appearance matching maintains correct ID

**Entry/Exit:**

- Entry: New track created, requires 3 consecutive detections
- Exit: Track deleted after 30 frames without match
- Re-entry: New ID if >30 frames, same ID if <30 frames

**Frame Drops:**

- Kalman prediction bridges gaps
- 30-frame buffer tolerates interruptions
- IoU + appearance features handle position shifts
- 10-frame velocity smoothing filters outliers

## Assumptions

1. **Video:** Consistent FPS, standard format, reasonable quality
2. **Camera:** Forward-facing dashcam, relatively stable
3. **Scene:** Highway with vehicles, objects large enough to detect
4. **Tracking:** Same ID maintained unless complete exit/re-entry
5. **Velocity:** Pixels/second (not real-world speed)

## Usage

pip install -r requirements.txt
python object_tracker.py

**Output:** `output_tracking.mp4` (annotated video), `tracking_data.csv` (frame_id, object_id, class, bbox, confidence, velocity)

## Results

**Test:** challenge-mle2.mp4 (960×540, 25 FPS, 681 frames)

- 65 unique tracks, 1,974 detections
- Longest track: 493 frames (19.7s)
- Average confidence: 60.4%

## OUTPUT File:-

- https://drive.google.com/file/d/1qILQA6BshnDbQjwKJN075-i-quw-5Yeq/view?usp=sharing
