# Store Traffic Monitoring System

A comprehensive system for monitoring store traffic using multiple CCTV cameras with face recognition, visitor tracking, and analytics.

## Features

- Multi-camera support with IP camera integration
- Real-time face detection and recognition
- Visitor tracking across different store zones
- Traffic analytics and heat map generation
- Dwell time analysis
- Peak hour detection
- Historical data tracking
- Real-time monitoring dashboard

## Components

1. **Camera Management (`camera_manager.py`)**
   - Handles multiple IP camera streams
   - Frame buffering and synchronization
   - Camera configuration management

2. **Face Recognition (`FaceRecognizer.py`)**
   - Face detection and recognition using InsightFace
   - FAISS-based fast similarity search
   - Visitor database management

3. **Traffic Analytics (`traffic_analytics.py`)**
   - Heat map generation
   - Visitor movement tracking
   - Zone-based analytics
   - Dwell time calculation
   - Traffic pattern analysis

4. **Store Monitoring System (`store_monitoring_system.py`)**
   - Main system integration
   - Real-time display
   - Multi-threaded processing
   - Analytics overlay

## Requirements

```bash
pip install -r requirements.txt
```

## CCTV Camera Setup

1. **Camera Requirements**
   - IP cameras with RTSP support
   - Supported models:
     * AE-IPPR03D (Dome camera)
     * AE-IPPR03B (Bullet camera)
   - NVR model: AE-PR16-NVR-S1
   - Camera specifications:
     * Resolution: 3MP (2048x1536)
     * Frame rate: 20fps
     * Video encoding: H.265
     * Protocol support: TCP/IP, UDP, HTTP, HTTPS, DHCP, RTSP, DDNS

2. **Network Requirements**
   - All cameras must be on the same network
   - Static IP addresses recommended
   - Open ports:
     * RTSP: 554 (default)
     * HTTP: 80
     * NVR: 8000

3. **Initial Setup**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the setup wizard
   ./setup_cameras.py
   ```
   The setup wizard will guide you through:
   - NVR configuration
   - Camera IP addresses and credentials
   - Zone assignments
   - Testing connections

4. **Camera Configuration**
   - The system supports two stream profiles:
     * Main stream: 3MP (2048x1536) @ 20fps
     * Sub stream: 2MP (1920x1080) @ 15fps
   - IR LED control: Auto/Manual/Off
   - Motion detection with configurable areas
   - Up to 5 people detection per frame

5. **Store Layout**
   - Define store zones in `zone_config.json`
   - Available zones:
     * entrance
     * checkout
     * electronics
     * clothing
     * groceries
     * home_goods
   - Each zone requires boundary coordinates

6. **Testing Cameras**
   ```bash
   # Test all configured cameras
   python test_camera_connection.py --all
   
   # Test specific camera
   python test_camera_connection.py --camera-id cam1 --ip 192.168.1.101 --username admin --password admin --display
   ```

7. **Troubleshooting**
   - Verify network connectivity with ping
   - Check RTSP URLs format:
     ```
     rtsp://{username}:{password}@{camera_ip}:{port}/stream1
     ```
   - Ensure camera firmware is up to date
   - Check camera is in the correct IP range
   - Verify port forwarding if accessing remotely

## Usage

1. Start the monitoring system:
```bash
python store_monitoring_system.py
```

2. Add cameras:
```python
system = StoreMonitoringSystem()
system.add_camera("cam1", "rtsp://camera1_url", "entrance", (0, 0))
system.add_camera("cam2", "rtsp://camera2_url", "checkout", (600, 400))
```

3. View Analytics:
   - Real-time analytics are displayed on the monitoring dashboard
   - Reports are saved in the `analytics` directory
   - Heat maps and traffic graphs are generated periodically

## Analytics Output

The system generates several types of analytics:

1. **Traffic Reports**
   - JSON files with visitor statistics
   - Peak hour information
   - Zone-wise visitor counts

2. **Heat Maps**
   - Visual representation of visitor movement patterns
   - Zone activity intensity

3. **Traffic Graphs**
   - Hourly visitor trends
   - Historical traffic patterns

## Data Storage

1. **Visitor Data**
   - Stored in `Visitors.csv`
   - Contains visitor IDs, timestamps, and confidence scores

2. **Face Embeddings**
   - Stored in `face_embeddings.pkl`
   - Used for face recognition and matching

3. **Analytics Data**
   - Reports saved in `analytics` directory
   - Heat maps and graphs saved as images

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- Sufficient storage for analytics data
- Network connectivity for IP cameras

## Notes

- The system automatically creates required directories and configuration files if they don't exist
- Face recognition models are downloaded automatically on first run
- Analytics are saved every 5 minutes by default (configurable)
- The display supports up to 3 cameras per row for optimal viewing

## Troubleshooting

1. **Camera Connection Issues**
   - Check RTSP URLs
   - Verify network connectivity
   - Ensure camera credentials are correct

2. **Performance Issues**
   - Adjust frame skip rate in camera manager
   - Reduce number of concurrent cameras
   - Use GPU acceleration if available

3. **Storage Issues**
   - Configure data retention period
   - Regular cleanup of old analytics files
   - Monitor disk space usage

## License

This project is licensed under the MIT License - see the LICENSE file for details.
