# Eye Blink Pattern Detection System 👁️

A real-time eye blink detection and analysis system using computer vision and machine learning. This system monitors eye blink patterns to assess cognitive states, detect fatigue, and identify potential issues like eye strain or stress through comprehensive JSON logging.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.0+-green.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.30+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📸 Demo

<div align="center">
  <img src="https://raw.githubusercontent.com/NeilAlvn/NeilAlvn.github.io/main/Blink.png" alt="Eye Blink Detection Demo" width="800">
  <p><i>Real-time eye blink detection with pattern analysis overlay</i></p>
</div>

## 🌟 Features

### Core Functionality
- **Real-time Face Detection**: Uses MediaPipe's Face Landmarker for accurate facial landmark detection
- **Eye Aspect Ratio (EAR) Calculation**: Monitors eye openness using geometric analysis
- **Blink Detection**: Identifies and counts individual blinks with high accuracy
- **Pattern Analysis**: Evaluates blinking patterns to assess cognitive and physical states

### Advanced Detection
- **Prolonged Eye Closure Detection**: Alerts when eyes remain closed for >3 seconds
- **Asymmetrical Blinking**: Detects imbalances between left and right eye movements
- **Rapid Blinking Bursts**: Identifies stress indicators (8+ blinks in 10 seconds)
- **Squinting Detection**: Monitors frequent squinting that may indicate eye strain

### Comprehensive Logging
- **JSON-based Data Export**: All events saved in structured, analyzable format
- **Three-tier Logging System**:
  - `blinks_*.json` - Individual blink events with timestamps
  - `patterns_*.json` - Pattern snapshots every 10 seconds
  - `alerts_*.json` - All detected anomalies and warnings
- **Session Summaries**: Statistical analysis of entire monitoring sessions

### Pattern Interpretation
The system interprets blink rates to assess user state:

| Blink Rate (per minute) | Pattern | Interpretation |
|-------------------------|---------|----------------|
| 15-20 | Normal | Alert, comfortable, normal cognitive state |
| 10-15 | Slightly reduced | High concentration, deep focus |
| 20-30 | Slightly increased | Thinking, processing information |
| >30 | Excessive | Anxiety, cognitive overload, stress |
| <10 | Severely reduced | Screen fatigue, eye strain, dissociation |

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Operating System: Windows, macOS, or Linux

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/eye-blink-detection.git
cd eye-blink-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python blink_detector.py
```

The application will automatically download the required MediaPipe Face Landmarker model on first run (~10MB).

## 📦 Dependencies

```
opencv-python>=4.8.0
numpy>=1.24.0
mediapipe>=0.10.30
```

## 💻 Usage

### Basic Operation

1. **Start the Application**:
   ```bash
   python blink_detector.py
   ```

2. **Position Yourself**: 
   - Sit comfortably in front of your webcam
   - Ensure good lighting on your face
   - Look naturally at the camera

3. **Monitor in Real-time**:
   - The application displays live video with overlay information
   - View current EAR (Eye Aspect Ratio) values
   - See blink count and rate updates
   - Receive alerts for unusual patterns

4. **Keyboard Controls**:
   - `Q` - Quit and save all logs
   - `R` - Reset counters (starts new monitoring session)

### Output Files

After each session, three JSON files are created in the `blink_logs/` directory:

#### 1. Blink Events (`blinks_YYYYMMDD_HHMMSS.json`)
Contains every detected blink with detailed metrics:
```json
{
  "event_type": "blink",
  "timestamp": "2024-01-31T14:30:45.123456",
  "elapsed_seconds": 125.45,
  "blink_number": 42,
  "ear_left": 0.185,
  "ear_right": 0.188,
  "avg_ear": 0.1865
}
```

#### 2. Pattern Snapshots (`patterns_YYYYMMDD_HHMMSS.json`)
Periodic assessments (every 10 seconds):
```json
{
  "timestamp": "2024-01-31T14:30:50.000000",
  "elapsed_seconds": 130.0,
  "blink_rate": 18,
  "total_blinks": 39,
  "pattern": "Normal blinking rate",
  "assessment": "Positive",
  "interpretation": "Alert, comfortable, normal cognitive state",
  "rapid_burst_detected": false,
  "squinting_detected": false
}
```

#### 3. Alerts (`alerts_YYYYMMDD_HHMMSS.json`)
All detected anomalies:
```json
{
  "alert_type": "prolonged_closure",
  "timestamp": "2024-01-31T14:31:15.789012",
  "elapsed_seconds": 155.78,
  "details": {
    "duration_seconds": 3.45
  }
}
```

## 🔬 Technical Details

### Eye Aspect Ratio (EAR)
The system uses the Eye Aspect Ratio formula to determine eye openness:

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Where p1-p6 are specific eye landmarks detected by MediaPipe.

### Detection Thresholds
- **EAR Threshold**: 0.21 (eyes considered closed below this)
- **Consecutive Frames**: 2 (blink confirmed after 2 frames)
- **Prolonged Closure**: 3.0 seconds
- **Rapid Burst**: 8+ blinks in 10 seconds
- **Asymmetry Threshold**: 0.08 EAR difference between eyes

### Face Landmarks Used
- **Left Eye**: Indices [33, 160, 158, 133, 153, 144]
- **Right Eye**: Indices [362, 385, 387, 263, 373, 380]

## 📊 Use Cases

### 1. **Productivity Monitoring**
- Track focus and concentration levels during work
- Identify optimal working periods
- Detect when breaks are needed

### 2. **Health & Wellness**
- Monitor screen time effects on eye health
- Detect early signs of digital eye strain
- Track recovery patterns after breaks

### 3. **Research & Development**
- Collect data for cognitive load studies
- Analyze user engagement with content
- Evaluate interface design effectiveness

### 4. **Gaming & Streaming**
- Monitor player engagement and immersion
- Detect fatigue during long sessions
- Analyze viewer attention patterns

### 5. **Accessibility**
- Develop assistive technologies
- Create eye-blink based controls
- Detect attention and alertness levels

## 🛠️ Configuration

### Adjusting Sensitivity

You can modify detection parameters in the `BlinkPatternDetectorJSON` class:

```python
# In __init__ method:
self.EAR_THRESHOLD = 0.21          # Lower = more sensitive
self.CONSECUTIVE_FRAMES = 2         # Blink confirmation frames
self.CLOSURE_THRESHOLD = 3.0        # Prolonged closure (seconds)
self.snapshot_interval = 10         # Pattern snapshot frequency

# In detect_rapid_burst method:
recent_blinks = [t for t in self.blink_timestamps if current_time - t < 10]
if len(recent_blinks) >= 8:        # Rapid burst threshold

# In detect_squinting method:
squint_margin = 0.06               # Squinting detection sensitivity
return squinting_frames > 17       # Frames threshold
```

### Camera Settings

Modify camera parameters in the `main()` function:

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Resolution width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Resolution height
cap.set(cv2.CAP_PROP_FPS, 30)             # Frame rate
```

## 📈 Data Analysis

### Example: Loading and Analyzing Logs

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load blink events
with open('blink_logs/blinks_20240131_143000.json', 'r') as f:
    blinks = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(blinks)

# Plot blink rate over time
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df['blinks_per_minute'] = df.resample('1T').size()

plt.figure(figsize=(12, 6))
plt.plot(df['blinks_per_minute'])
plt.title('Blink Rate Over Time')
plt.xlabel('Time')
plt.ylabel('Blinks per Minute')
plt.show()
```

## 🐛 Troubleshooting

### Camera Not Detected
```bash
# Test camera access
python -c "import cv2; print('Camera working!' if cv2.VideoCapture(0).isOpened() else 'Camera failed')"
```

### MediaPipe Import Error
```bash
# Reinstall MediaPipe
pip uninstall mediapipe
pip install mediapipe>=0.10.30
```

### Model Download Fails
- Ensure internet connection is stable
- Check firewall settings
- Manually download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- Place in project root directory

### Low FPS / Performance Issues
- Reduce camera resolution
- Close other applications
- Ensure adequate lighting (reduces processing time)
- Check CPU usage

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit Your Changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Update README.md with new features
- Test thoroughly before submitting

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** - Google's framework for building multimodal ML pipelines
- **OpenCV** - Computer vision library
- **Eye Aspect Ratio (EAR)** - Based on research by Soukupová and Čech (2016)

## 📧 Contact

Project Link: [https://github.com/NeilAlvn/eye-blink-detection](https://github.com/yourusername/eye-blink-detection)

## 🔮 Future Enhancements

- [ ] Multi-face tracking support
- [ ] Real-time data visualization dashboard
- [ ] Machine learning model for pattern classification
- [ ] Mobile app version
- [ ] Cloud-based data storage and analysis
- [ ] Integration with productivity tools
- [ ] Customizable alert notifications
- [ ] Export to CSV/Excel formats
- [ ] Statistical analysis reports
- [ ] Eye health recommendations

## 📚 References

1. Soukupová, T., & Čech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks. 21st Computer Vision Winter Workshop.
2. MediaPipe Face Landmark Detection: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
3. Eye Aspect Ratio for Blink Detection: Various research papers on drowsiness detection

---

⭐ **Star this repository** if you find it useful!

**Made with ❤️ by Neil Alvin Medallon**
