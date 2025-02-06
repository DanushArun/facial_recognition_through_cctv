#!/usr/bin/env python3
import cv2
import time
import json
import os
from camera_manager import CameraManager
from store_monitoring_system import StoreMonitoringSystem
import argparse
from typing import Dict, List
import socket

def test_network_connectivity(ip: str, port: int) -> bool:
    """Test if a network endpoint is reachable."""
    try:
        sock = socket.create_connection((ip, port), timeout=2)
        sock.close()
        return True
    except (socket.timeout, socket.error):
        return False

def validate_camera_config(config_file: str = "camera_config.json") -> List[Dict]:
    """Validate camera configuration file and settings."""
    issues = []
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check NVR settings
        nvr = config.get('nvr_settings', {})
        if not nvr:
            issues.append("Missing NVR settings")
        else:
            if not test_network_connectivity(nvr['ip'], nvr['port']):
                issues.append(f"NVR not reachable at {nvr['ip']}:{nvr['port']}")
        
        # Check camera configurations
        cameras = config.get('cameras', [])
        if not cameras:
            issues.append("No cameras configured")
        
        for camera in cameras:
            # Check required fields
            required_fields = ['id', 'ip', 'port', 'username', 'password', 'zone']
            missing_fields = [field for field in required_fields if field not in camera]
            if missing_fields:
                issues.append(f"Camera {camera.get('id', 'unknown')}: Missing fields: {missing_fields}")
            
            # Check network connectivity
            if 'ip' in camera and 'port' in camera:
                if not test_network_connectivity(camera['ip'], camera['port']):
                    issues.append(f"Camera {camera.get('id', 'unknown')} not reachable at {camera['ip']}:{camera['port']}")
        
        return issues
    
    except FileNotFoundError:
        return ["Configuration file not found"]
    except json.JSONDecodeError:
        return ["Invalid JSON in configuration file"]
    except Exception as e:
        return [f"Unexpected error: {str(e)}"]

def test_camera_streams(manager: CameraManager, duration: int = 10) -> Dict[str, bool]:
    """Test camera streams for a specified duration."""
    results = {}
    start_time = time.time()
    frame_counts = {cam_id: 0 for cam_id in manager.cameras.keys()}
    
    print(f"\nTesting camera streams for {duration} seconds...")
    
    try:
        while time.time() - start_time < duration:
            frames = manager.get_frames()
            for camera_id, frame in frames.items():
                if frame is not None:
                    frame_counts[camera_id] += 1
                
                # Display frames with status overlay
                if frame is not None:
                    status_text = f"Frames: {frame_counts[camera_id]}"
                    cv2.putText(frame, status_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(f"Camera {camera_id}", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Calculate results
        for camera_id, count in frame_counts.items():
            # Consider test successful if we got at least 1 frame per second
            results[camera_id] = count >= duration
        
    finally:
        cv2.destroyAllWindows()
    
    return results

def test_face_recognition(system: StoreMonitoringSystem, duration: int = 10) -> bool:
    """Test face recognition system."""
    print(f"\nTesting face recognition for {duration} seconds...")
    
    start_time = time.time()
    face_detected = False
    
    try:
        while time.time() - start_time < duration:
            frames = system.camera_manager.get_frames()
            for camera_id, frame in frames.items():
                if frame is not None:
                    # Process frame with face recognition
                    processed_frame = system.face_recognizer.process_frame(frame)
                    
                    # Check if any faces were detected
                    faces = system.face_recognizer.app.get(frame)
                    if len(faces) > 0:
                        face_detected = True
                    
                    # Display frame
                    cv2.imshow(f"Face Recognition Test - Camera {camera_id}", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cv2.destroyAllWindows()
    
    return face_detected

def main():
    parser = argparse.ArgumentParser(description='Test system integration')
    parser.add_argument('--duration', type=int, default=10,
                      help='Duration for each test in seconds')
    parser.add_argument('--config', type=str, default='camera_config.json',
                      help='Path to camera configuration file')
    args = parser.parse_args()
    
    print("Starting system integration tests...")
    
    # Step 1: Validate configuration
    print("\nValidating configuration...")
    issues = validate_camera_config(args.config)
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"- {issue}")
        if input("\nContinue with tests? (y/n): ").lower() != 'y':
            return
    else:
        print("Configuration validation passed")
    
    # Step 2: Test camera streams
    print("\nTesting camera streams...")
    manager = CameraManager(args.config)
    manager.start_cameras()
    
    stream_results = test_camera_streams(manager, args.duration)
    print("\nCamera stream test results:")
    for camera_id, success in stream_results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"Camera {camera_id}: {status}")
    
    # Step 3: Test face recognition
    print("\nTesting face recognition system...")
    system = StoreMonitoringSystem()
    face_recognition_success = test_face_recognition(system, args.duration)
    print(f"Face Recognition Test: {'✓ PASS' if face_recognition_success else '✗ FAIL'}")
    
    # Step 4: Test analytics
    print("\nTesting analytics system...")
    try:
        # Generate test analytics
        system.analytics.save_analytics_report()
        print("Analytics Test: ✓ PASS")
    except Exception as e:
        print(f"Analytics Test: ✗ FAIL - {str(e)}")
    
    # Cleanup
    manager.stop_cameras()
    
    print("\nTest Summary:")
    print("-" * 40)
    print(f"Configuration: {'✓ PASS' if not issues else '✗ FAIL'}")
    print(f"Camera Streams: {'✓ PASS' if all(stream_results.values()) else '✗ FAIL'}")
    print(f"Face Recognition: {'✓ PASS' if face_recognition_success else '✗ FAIL'}")
    print(f"Analytics: {'✓ PASS' if os.path.exists('analytics') else '✗ FAIL'}")

if __name__ == "__main__":
    main()
