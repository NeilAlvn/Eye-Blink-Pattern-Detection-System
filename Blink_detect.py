"""
Eye Blink Pattern Detection System with JSON Logging
Logs all detections, patterns, and summaries to JSON files
"""

import cv2
import numpy as np
import time
import json
from collections import deque
from datetime import datetime
import os
import urllib.request

# Import MediaPipe
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    try:
        from mediapipe.python._framework_bindings import image as mp_image # type: ignore
        Image = mp_image.Image
        ImageFormat = mp_image.ImageFormat
    except:
        try:
            from mediapipe import Image, ImageFormat
        except:
            Image = None
            ImageFormat = None
    print("✓ MediaPipe imported successfully")
except ImportError as e:
    print(f"ERROR: Cannot import MediaPipe tasks: {e}")
    print("Install with: pip install mediapipe>=0.10.30")
    exit(1)


class BlinkPatternDetectorJSON:
    def __init__(self, output_dir="blink_logs"):
        """Initialize the blink detector with JSON logging"""
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        
        # Download model if needed
        self.model_path = 'face_landmarker.task'
        self._ensure_model_exists()
        
        # Initialize MediaPipe Face Landmarker
        print("Initializing face detector...")
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.detector = vision.FaceLandmarker.create_from_options(options)
        print("✓ Face detector initialized")
        
        # Eye landmarks indices
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        
        # Thresholds
        self.EAR_THRESHOLD = 0.21
        self.CONSECUTIVE_FRAMES = 2
        self.CLOSURE_THRESHOLD = 3.0  # Increased from 2.0 to 3.0 seconds
        
        # State tracking
        self.frame_counter = 0
        self.total_blinks = 0
        self.blink_timestamps = deque(maxlen=1000)
        self.ear_history = deque(maxlen=30)
        self.eye_closed_start = None
        self.last_minute_blinks = deque(maxlen=60)
        
        # Pattern detection
        self.start_time = time.time()
        
        # JSON logging
        self.blink_events = []  # Store all blink events
        self.pattern_snapshots = []  # Store pattern assessments over time
        self.alerts = []  # Store all alerts
        self.last_snapshot_time = time.time()
        self.snapshot_interval = 10  # Take snapshot every 10 seconds
        
        print("✓ Detector ready!\n")
        print(f"Session ID: {self.session_id}")
        print(f"Output directory: {self.output_dir}\n")
    
    def _ensure_model_exists(self):
        """Download the face landmarker model if it doesn't exist"""
        if os.path.exists(self.model_path):
            print(f"✓ Model found: {self.model_path}")
            return
        
        print("Downloading face landmarker model (first time only)...")
        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
        
        try:
            urllib.request.urlretrieve(url, self.model_path)
            print(f"✓ Model downloaded successfully: {self.model_path}")
        except Exception as e:
            print(f"\n❌ ERROR downloading model: {e}")
            exit(1)
    
    def calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio (EAR)"""
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        if h == 0:
            return 0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def extract_eye_landmarks(self, face_landmarks, indices, img_width, img_height):
        """Extract specific eye landmark coordinates"""
        points = []
        
        for idx in indices:
            landmark = face_landmarks[idx]
            x = int(landmark.x * img_width)
            y = int(landmark.y * img_height)
            points.append(np.array([x, y]))
        
        return np.array(points)
    
    def detect_blinks(self, frame):
        """Process a frame and detect blinks"""
        
        h, w = frame.shape[:2]
        
        current_status = {
            'ear_left': 0,
            'ear_right': 0,
            'avg_ear': 0,
            'is_blinking': False,
            'prolonged_closure': False,
            'asymmetrical': False,
            'face_detected': False
        }
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        try:
            if Image is not None and ImageFormat is not None:
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            else:
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        except Exception as e:
            print(f"Error creating image: {e}")
            return frame, current_status
        
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            current_status['face_detected'] = True
            face_landmarks = detection_result.face_landmarks[0]
            
            left_eye = self.extract_eye_landmarks(face_landmarks, self.LEFT_EYE_INDICES, w, h)
            right_eye = self.extract_eye_landmarks(face_landmarks, self.RIGHT_EYE_INDICES, w, h)
            
            ear_left = self.calculate_ear(left_eye)
            ear_right = self.calculate_ear(right_eye)
            avg_ear = (ear_left + ear_right) / 2.0
            
            current_status['ear_left'] = ear_left
            current_status['ear_right'] = ear_right
            current_status['avg_ear'] = avg_ear
            
            self.ear_history.append(avg_ear)
            
            # Check for asymmetrical blinking - increased threshold from 0.05 to 0.08
            ear_diff = abs(ear_left - ear_right)
            if ear_diff > 0.08:
                current_status['asymmetrical'] = True
                self._log_alert("asymmetrical_blinking", {
                    "ear_left": float(ear_left),
                    "ear_right": float(ear_right),
                    "difference": float(ear_diff)
                })
            
            # Detect blink
            if avg_ear < self.EAR_THRESHOLD:
                self.frame_counter += 1
                
                if self.eye_closed_start is None:
                    self.eye_closed_start = time.time()
                else:
                    closure_duration = time.time() - self.eye_closed_start
                    if closure_duration > self.CLOSURE_THRESHOLD:
                        current_status['prolonged_closure'] = True
                        self._log_alert("prolonged_closure", {
                            "duration_seconds": float(closure_duration)
                        })
            else:
                if self.frame_counter >= self.CONSECUTIVE_FRAMES:
                    self.total_blinks += 1
                    current_time = time.time()
                    self.blink_timestamps.append(current_time)
                    self.last_minute_blinks.append(current_time)
                    current_status['is_blinking'] = True
                    
                    # Log blink event
                    self._log_blink_event(ear_left, ear_right, avg_ear)
                
                self.frame_counter = 0
                self.eye_closed_start = None
            
            # Draw eye landmarks
            for point in left_eye:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
            for point in right_eye:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
        
        return frame, current_status
    
    def calculate_blink_rate(self):
        """Calculate blinks per minute"""
        current_time = time.time()
        
        while self.last_minute_blinks and (current_time - self.last_minute_blinks[0]) > 60:
            self.last_minute_blinks.popleft()
        
        return len(self.last_minute_blinks)
    
    def detect_rapid_burst(self):
        """Detect rapid blinking burst - now requires 8+ blinks in 10 seconds"""
        if len(self.blink_timestamps) < 8:  # Increased from 5 to 8
            return False
        
        current_time = time.time()
        recent_blinks = [t for t in self.blink_timestamps if current_time - t < 10]
        
        if len(recent_blinks) >= 8:  # Increased from 5 to 8
            return True
        return False
    
    def detect_squinting(self):
        """Detect frequent squinting - balanced sensitivity"""
        if len(self.ear_history) < 10:
            return False
        
        # Balanced margin - not too sensitive, not too strict
        squint_margin = 0.06  # Between old (0.05) and strict (0.07)
        squinting_frames = sum(
            1 for ear in list(self.ear_history)[-30:] 
            if self.EAR_THRESHOLD < ear < (self.EAR_THRESHOLD + squint_margin)
        )
        
        # Require 18 out of 30 frames (60%) - balanced threshold
        return squinting_frames > 17
    
    def assess_pattern(self):
        """Assess current blinking pattern"""
        blink_rate = self.calculate_blink_rate()
        
        if 15 <= blink_rate <= 20:
            pattern = "Normal blinking rate"
            assessment = "Positive"
            interpretation = "Alert, comfortable, normal cognitive state"
        elif 10 <= blink_rate < 15:
            pattern = "Slightly reduced blinking"
            assessment = "Neutral"
            interpretation = "High concentration, deep focus on content"
        elif 20 < blink_rate <= 30:
            pattern = "Slightly increased blinking"
            assessment = "Neutral"
            interpretation = "Thinking, processing information"
        elif blink_rate > 30:
            pattern = "Excessive blinking"
            assessment = "Negative"
            interpretation = "Anxiety, cognitive overload, stress"
        elif blink_rate < 10:
            pattern = "Severely reduced blinking"
            assessment = "Negative"
            interpretation = "Screen fatigue, eye strain, dissociation"
        else:
            pattern = "Unknown"
            assessment = "Neutral"
            interpretation = "Insufficient data"
        
        rapid_burst = self.detect_rapid_burst()
        squinting = self.detect_squinting()
        
        # Log rapid burst alert
        if rapid_burst:
            self._log_alert("rapid_blinking_burst", {
                "blinks_in_10_seconds": len([t for t in self.blink_timestamps if time.time() - t < 10])
            })
        
        # Log squinting alert
        if squinting:
            squint_margin = 0.06  # Match the detection threshold
            self._log_alert("frequent_squinting", {
                "squinting_frames": sum(1 for ear in list(self.ear_history)[-30:] 
                                       if self.EAR_THRESHOLD < ear < (self.EAR_THRESHOLD + squint_margin))
            })
        
        return {
            'blink_rate': blink_rate,
            'pattern': pattern,
            'assessment': assessment,
            'interpretation': interpretation,
            'total_blinks': self.total_blinks,
            'rapid_burst': rapid_burst,
            'squinting': squinting
        }
    
    def _log_blink_event(self, ear_left, ear_right, avg_ear):
        """Log individual blink event"""
        event = {
            "event_type": "blink",
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": float(time.time() - self.start_time),
            "blink_number": self.total_blinks,
            "ear_left": float(ear_left),
            "ear_right": float(ear_right),
            "avg_ear": float(avg_ear)
        }
        self.blink_events.append(event)
    
    def _log_alert(self, alert_type, details):
        """Log an alert"""
        # Avoid duplicate alerts in quick succession - increased from 5 to 30 seconds
        if self.alerts:
            for alert in reversed(self.alerts[-10:]):  # Check last 10 alerts
                if alert["alert_type"] == alert_type:
                    time_since_last = time.time() - self.start_time - alert["elapsed_seconds"]
                    if time_since_last < 30:  # Only log same alert once per 30 seconds
                        return  # Skip logging this duplicate
                    break
        
        alert = {
            "alert_type": alert_type,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": float(time.time() - self.start_time),
            "details": details
        }
        self.alerts.append(alert)
        
        # Print to console so user knows alert was logged
        elapsed_mins = alert["elapsed_seconds"] / 60
        print(f"🚨 ALERT [{elapsed_mins:.1f}m]: {alert_type.replace('_', ' ').title()}")
    
    def take_pattern_snapshot(self, pattern_info):
        """Take a snapshot of current pattern"""
        current_time = time.time()
        
        if current_time - self.last_snapshot_time >= self.snapshot_interval:
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": float(current_time - self.start_time),
                "blink_rate": pattern_info['blink_rate'],
                "total_blinks": pattern_info['total_blinks'],
                "pattern": pattern_info['pattern'],
                "assessment": pattern_info['assessment'],
                "interpretation": pattern_info['interpretation'],
                "rapid_burst_detected": pattern_info['rapid_burst'],
                "squinting_detected": pattern_info['squinting']
            }
            self.pattern_snapshots.append(snapshot)
            self.last_snapshot_time = current_time
    
    def generate_summary(self):
        """Generate session summary"""
        elapsed_time = time.time() - self.start_time
        
        # Calculate statistics
        total_alerts = len(self.alerts)
        alert_types = {}
        for alert in self.alerts:
            alert_type = alert['alert_type']
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
        
        # Pattern distribution
        pattern_distribution = {}
        for snapshot in self.pattern_snapshots:
            pattern = snapshot['pattern']
            pattern_distribution[pattern] = pattern_distribution.get(pattern, 0) + 1
        
        # Average blink rate
        if self.pattern_snapshots:
            avg_blink_rate = sum(s['blink_rate'] for s in self.pattern_snapshots) / len(self.pattern_snapshots)
        else:
            avg_blink_rate = 0
        
        summary = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "duration_seconds": float(elapsed_time),
            "duration_minutes": float(elapsed_time / 60),
            "total_blinks": self.total_blinks,
            "average_blink_rate": float(avg_blink_rate),
            "total_alerts": total_alerts,
            "alerts_by_type": alert_types,
            "pattern_distribution": pattern_distribution,
            "snapshots_taken": len(self.pattern_snapshots),
            "blink_events_logged": len(self.blink_events)
        }
        
        return summary
    
    def save_json_logs(self):
        """Save logs to 3 JSON files"""
        
        # Generate summary
        summary = self.generate_summary()
        
        # 1. Save blink events with timestamps
        events_file = os.path.join(self.output_dir, f"blinks_{self.session_id}.json")
        with open(events_file, 'w') as f:
            json.dump(self.blink_events, f, indent=2)
        print(f"✓ Blink events saved: {events_file}")
        
        # 2. Save pattern snapshots with timestamps
        snapshots_file = os.path.join(self.output_dir, f"patterns_{self.session_id}.json")
        with open(snapshots_file, 'w') as f:
            json.dump(self.pattern_snapshots, f, indent=2)
        print(f"✓ Pattern history saved: {snapshots_file}")
        
        # 3. Save alerts with timestamps
        alerts_file = os.path.join(self.output_dir, f"alerts_{self.session_id}.json")
        with open(alerts_file, 'w') as f:
            json.dump(self.alerts, f, indent=2)
        print(f"✓ Alerts saved: {alerts_file}")
        
        # Also save summary at the top of each file for context
        print(f"\n📊 Session Summary:")
        print(f"   Duration: {summary['duration_minutes']:.1f} minutes")
        print(f"   Total Blinks: {summary['total_blinks']}")
        print(f"   Avg Blink Rate: {summary['average_blink_rate']:.1f}/min")
        print(f"   Total Alerts: {summary['total_alerts']}")
        
        return events_file, snapshots_file, alerts_file
    
    def draw_info(self, frame, current_status, pattern_info):
        """Draw information overlay on the frame"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (min(550, w - 10), 320), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "Eye Blink Pattern Monitor (JSON Logging)", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if not current_status['face_detected']:
            cv2.putText(frame, "No face detected - please look at camera", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            return frame
        
        y = 75
        cv2.putText(frame, f"EAR: {current_status['avg_ear']:.3f}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 25
        cv2.putText(frame, f"Total Blinks: {pattern_info['total_blinks']}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 25
        cv2.putText(frame, f"Blink Rate: {pattern_info['blink_rate']}/min", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 25
        elapsed = time.time() - self.start_time
        cv2.putText(frame, f"Time: {int(elapsed//60)}m {int(elapsed%60)}s", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 30
        cv2.putText(frame, f"Pattern: {pattern_info['pattern']}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        colors = {
            'Positive': (0, 255, 0),
            'Neutral': (0, 255, 255),
            'Negative': (0, 100, 255)
        }
        color = colors.get(pattern_info['assessment'], (255, 255, 255))
        
        y += 25
        cv2.putText(frame, f"Assessment: {pattern_info['assessment']}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        y += 30
        interpretation = pattern_info['interpretation']
        if len(interpretation) > 45:
            words = interpretation.split()
            mid = len(words) // 2
            line1 = ' '.join(words[:mid])
            line2 = ' '.join(words[mid:])
            cv2.putText(frame, line1, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y += 18
            cv2.putText(frame, line2, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.putText(frame, interpretation, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y += 25
        cv2.putText(frame, f"Logged: {len(self.blink_events)} blinks, {len(self.alerts)} alerts", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        
        y += 25
        if current_status['prolonged_closure']:
            cv2.putText(frame, "WARNING: Prolonged eye closure", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif current_status['asymmetrical']:
            cv2.putText(frame, "ALERT: Asymmetrical blinking", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        elif pattern_info['rapid_burst']:
            cv2.putText(frame, "ALERT: Rapid blinking burst", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        elif pattern_info['squinting']:
            cv2.putText(frame, "NOTICE: Frequent squinting", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return frame


def main():
    """Main function"""
    print("\n" + "="*60)
    print("  Eye Blink Pattern Detection System - JSON Logger")
    print("="*60 + "\n")
    
    try:
        detector = BlinkPatternDetectorJSON()
    except Exception as e:
        print(f"\n❌ ERROR initializing detector: {e}")
        return
    
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera opened successfully!\n")
    print("Instructions:")
    print("  • Look at the camera naturally")
    print("  • Press 'q' to quit and save logs")
    print("  • Press 'r' to reset counters")
    print("\nStarting detection...\n")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame, current_status = detector.detect_blinks(frame)
            pattern_info = detector.assess_pattern()
            
            # Take periodic snapshots
            detector.take_pattern_snapshot(pattern_info)
            
            frame = detector.draw_info(frame, current_status, pattern_info)
            cv2.imshow('Eye Blink Pattern Monitor - Press Q to quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                print("\n🔄 Resetting counters...")
                detector.total_blinks = 0
                detector.blink_timestamps.clear()
                detector.last_minute_blinks.clear()
                detector.start_time = time.time()
                print("✓ Counters reset\n")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        
        # Save JSON logs
        print("\nSaving JSON logs...")
        files = detector.save_json_logs()
        
        # Print summary
        summary = detector.generate_summary()
        print("\n" + "="*60)
        print("  Session Summary")
        print("="*60)
        print(f"Session ID: {summary['session_id']}")
        print(f"Duration: {summary['duration_minutes']:.2f} minutes")
        print(f"Total blinks: {summary['total_blinks']}")
        print(f"Average blink rate: {summary['average_blink_rate']:.1f} blinks/min")
        print(f"Total alerts: {summary['total_alerts']}")
        if summary['alerts_by_type']:
            print("\nAlerts by type:")
            for alert_type, count in summary['alerts_by_type'].items():
                print(f"  - {alert_type}: {count}")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()