# File: drowning_detector.py
import cv2
import torch
from datetime import datetime, timedelta

class DrowningDetector:
    def __init__(self):
        # Load trained model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                   path='best.pt')  # Your trained weights
        
        # Initialize alert system
        self.alert_buffer = {}
        self.ALERT_DURATION = timedelta(seconds=15)
        
        # Thresholds (adjust based on validation)
        self.DROWNING_CONFIDENCE = 0.7
        self.MIN_ALERT_DURATION = 15  # seconds

    def process_frame(self, frame):
        # YOLOv5 inference
        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        
        current_time = datetime.now()
        active_alerts = []

        # Process detections
        for _, det in detections.iterrows():
            if det['name'] == 'drowning' and det['confidence'] > self.DROWNING_CONFIDENCE:
                # Track detection time
                bbox_id = f"{det['xmin']}-{det['ymin']}-{det['xmax']}-{det['ymax']}"
                
                if bbox_id not in self.alert_buffer:
                    self.alert_buffer[bbox_id] = current_time
                else:
                    duration = (current_time - self.alert_buffer[bbox_id]).seconds
                    if duration >= self.MIN_ALERT_DURATION:
                        active_alerts.append((
                            int(det['xmin']), 
                            int(det['ymin']),
                            int(det['xmax']),
                            int(det['ymax'])
                        ))

        # Visualize results
        for box in active_alerts:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(frame, 'DROWNING ALERT!', (box[0], box[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
        return frame, len(active_alerts) > 0

# Usage example
if __name__ == "__main__":
    detector = DrowningDetector()
    
    # For webcam input
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, alert_status = detector.process_frame(frame)
        
        # Integrate with hardware systems
        if alert_status:
            # Trigger Arduino actions here
            print("ACTIVATING MECHANISM!")
            
        cv2.imshow('Drowning Detection', processed_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()