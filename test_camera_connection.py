import cv2
import time
from camera_manager import CameraManager
import argparse
import json

def test_camera_connection(camera_id: str, ip: str, username: str, password: str):
    """Test connection to a single camera."""
    # Create RTSP URL for main stream
    rtsp_url = f"rtsp://{username}:{password}@{ip}:554/stream1"
    
    print(f"\nTesting connection to camera {camera_id} at {ip}")
    print(f"RTSP URL: {rtsp_url}")
    
    # Try to connect
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Failed to connect to camera {camera_id}")
        return False
    
    # Try to read frames
    success_count = 0
    start_time = time.time()
    frames_to_test = 30  # Test 30 frames
    
    for _ in range(frames_to_test):
        ret, frame = cap.read()
        if ret:
            success_count += 1
    
    # Calculate success rate and FPS
    duration = time.time() - start_time
    success_rate = (success_count / frames_to_test) * 100
    fps = success_count / duration if duration > 0 else 0
    
    print(f"\nConnection Test Results for {camera_id}:")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average FPS: {fps:.1f}")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    cap.release()
    return success_rate > 80  # Consider test successful if >80% frames received

def test_all_cameras(config_file: str = "camera_config.json"):
    """Test all cameras in the configuration file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        results = []
        for camera in config['cameras']:
            success = test_camera_connection(
                camera['id'],
                camera['ip'],
                camera['username'],
                camera['password']
            )
            results.append((camera['id'], success))
        
        # Print summary
        print("\nTest Summary:")
        print("-" * 40)
        for camera_id, success in results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"Camera {camera_id}: {status}")
        
        # Check if all cameras passed
        all_passed = all(success for _, success in results)
        if all_passed:
            print("\nAll cameras connected successfully!")
        else:
            print("\nSome cameras failed to connect. Please check configurations.")
        
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found.")
    except json.JSONDecodeError:
        print(f"Error parsing configuration file {config_file}.")
    except Exception as e:
        print(f"Unexpected error: {e}")

def test_camera_display(camera_id: str, ip: str, username: str, password: str, duration: int = 10):
    """Test camera display for a specified duration."""
    rtsp_url = f"rtsp://{username}:{password}@{ip}:554/stream1"
    
    print(f"\nTesting video display for camera {camera_id}")
    print(f"Press 'q' to quit or wait {duration} seconds")
    
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Failed to open camera stream")
        return
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Add info overlay
        cv2.putText(frame, f"Camera: {camera_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(f"Camera {camera_id} Test", frame)
        
        # Check for quit command or timeout
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time() - start_time > duration:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test CCTV camera connections')
    parser.add_argument('--all', action='store_true', help='Test all cameras in config')
    parser.add_argument('--camera-id', help='Test specific camera ID')
    parser.add_argument('--ip', help='Camera IP address')
    parser.add_argument('--username', help='Camera username')
    parser.add_argument('--password', help='Camera password')
    parser.add_argument('--display', action='store_true', help='Show video feed')
    parser.add_argument('--duration', type=int, default=10, help='Display duration in seconds')
    
    args = parser.parse_args()
    
    if args.all:
        test_all_cameras()
    elif args.camera_id and args.ip and args.username and args.password:
        if args.display:
            test_camera_display(args.camera_id, args.ip, args.username, args.password, args.duration)
        else:
            test_camera_connection(args.camera_id, args.ip, args.username, args.password)
    else:
        print("Please provide either --all flag or all camera details (--camera-id, --ip, --username, --password)")
        print("\nExample usage:")
        print("Test all cameras:")
        print("  python test_camera_connection.py --all")
        print("\nTest specific camera:")
        print("  python test_camera_connection.py --camera-id cam1 --ip 192.168.1.101 --username admin --password admin")
        print("\nTest with video display:")
        print("  python test_camera_connection.py --camera-id cam1 --ip 192.168.1.101 --username admin --password admin --display")
