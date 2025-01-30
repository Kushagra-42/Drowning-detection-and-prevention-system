# File: sensor_fusion.py
import time
from collections import deque
from drowning_detector import DrowningDetector

class SafetyMonitor:
    def __init__(self):
        self.detector = DrowningDetector()
        self.health_buffer = deque(maxlen=15)  # 15-second window
        self.THRESHOLDS = {
            'hr': 50,
            'spo2': 90,
            'acc': 1.0
        }

    def update_health_data(self, hr, spo2, acc_x, acc_y, acc_z):
        # Calculate motion magnitude
        acc_magnitude = (acc_x**2 + acc_y**2 + acc_z**2)**0.5
        self.health_buffer.append({
            'timestamp': time.time(),
            'hr': hr,
            'spo2': spo2,
            'acc': acc_magnitude
        })

    def check_health_status(self):
        if len(self.health_buffer) < 15:
            return False

        # Check all values in buffer
        health_alert = all(
            (item['hr'] <= self.THRESHOLDS['hr'] and
             item['spo2'] <= self.THRESHOLDS['spo2'] and
             item['acc'] <= self.THRESHOLDS['acc'])
            for item in self.health_buffer
        )
        
        return health_alert

    def integrated_check(self, frame):
        # Computer vision check
        processed_frame, vision_alert = self.detector.process_frame(frame)
        
        # Health data check
        health_alert = self.check_health_status()
        
        # Decision logic
        if health_alert and vision_alert:
            return processed_frame, "PRIORITY 1: Full Alert"
        elif health_alert:
            return processed_frame, "PRIORITY 2: Health Alert"
        else:
            return processed_frame, "Normal"