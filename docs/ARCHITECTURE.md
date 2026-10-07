# Store Traffic Monitor — Architecture and implementation

This guide follows the tracked implementation. Proposed work is identified separately.

## The problem and the system boundary

Store traffic analysis involves stream availability, detections, identity matching and spatial
interpretation. This prototype combines those layers, but a visitor count or heat map is only as
reliable as the recognition and camera geometry feeding it.

## Processing path

```mermaid
flowchart LR
    N0["RTSP streams"]
    N1["Face matching"]
    N2["Zone metrics"]
    N3["Local reports"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
```

## End-to-end behavior

### 1. Configure a controlled camera

Supply authorized RTSP endpoints and zone boundaries. Review the connection diagnostics before
capturing data.

### 2. Inspect stream processing

The camera manager coordinates frames while the monitoring system processes and displays them.
Network reconnect behavior belongs to a different layer than recognition accuracy.

### 3. Review recognition records

Face matching can register visitors and save embeddings/CSV records. Existing artifacts are not a
labeled accuracy benchmark.

### 4. Check analytics output

Inspect dwell-time, zone and heat-map reports. Validate camera-to-store coordinates and false
matches before interpreting cross-camera behavior.

## Design choices and consequences

### Separate stream and recognition failures

A connected camera is not proof of valid face matching.

### Local artifact storage

CSV and pickle outputs need explicit trust, access and retention controls.

### Diagnostic tests are labeled

Live camera scripts are not isolated unit tests.

## Source entry points

### [camera_manager.py](../camera_manager.py)

- `CameraConfig` — Implementation entry; inspect source for its exact behavior.
- `CameraStream` — Implementation entry; inspect source for its exact behavior.
- `CameraManager` — Implementation entry; inspect source for its exact behavior.
- `__init__` — Implementation entry; inspect source for its exact behavior.
- `timeout` — Context manager for socket timeout.
- `connect` — Attempt to connect to the camera.

### [FaceRecognizer.py](../FaceRecognizer.py)

- `StoreVisitorRecognition` — Implementation entry; inspect source for its exact behavior.
- `__init__` — Implementation entry; inspect source for its exact behavior.
- `update_visitor_record` — Update visitor record in CSV
- `preprocess_frame` — Enhanced frame preprocessing for CCTV footage
- `preprocess_face` — Enhanced face preprocessing
- `process_frame` — Enhanced frame processing with attendance tracking and debug info

### [store_monitoring_system.py](../store_monitoring_system.py)

- `StoreMonitoringSystem` — Implementation entry; inspect source for its exact behavior.
- `__init__` — Implementation entry; inspect source for its exact behavior.
- `start` — Start the monitoring system.
- `stop` — Stop the monitoring system.
- `_process_frames` — Process frames from all cameras in a separate thread.
- `_display_feeds` — Display all camera feeds in a grid layout.

### [traffic_analytics.py](../traffic_analytics.py)

- `VisitorMetrics` — Implementation entry; inspect source for its exact behavior.
- `TrafficAnalytics` — Implementation entry; inspect source for its exact behavior.
- `__init__` — Implementation entry; inspect source for its exact behavior.
- `load_zone_config` — Load store zone configuration.
- `create_default_zone_config` — Create a default zone configuration file.
- `update_visitor_position` — Update visitor position and related metrics.

### [test_system_integration.py](../test_system_integration.py)

- `test_network_connectivity` — Test if a network endpoint is reachable.
- `validate_camera_config` — Validate camera configuration file and settings.
- `test_camera_streams` — Test camera streams for a specified duration.
- `test_face_recognition` — Test face recognition system.
- `main` — Implementation entry; inspect source for its exact behavior.

## Implementation state

| State | Evidence boundary |
| --- | --- |
| Present | Stream management and recognition source |
| Present | Visitor metrics, report and heat-map generation |
| Requires hardware | Live camera and integration diagnostics |
| Not validated | Identity accuracy, calibration, privacy or production readiness |

“Present” means tracked source or assets exist. It does not mean a production or domain
validation has passed. See [Evaluation](EVALUATION.md) for reproducible checks and limits.
