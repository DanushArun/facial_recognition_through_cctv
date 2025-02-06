import cv2
import threading
import queue
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import ipaddress
import socket
from contextlib import contextmanager

@dataclass
class CameraConfig:
    camera_id: str
    rtsp_url: str
    zone: str
    position: Tuple[int, int]  # x, y coordinates in store layout

class CameraStream(threading.Thread):
    def __init__(self, camera_id: str, rtsp_url: str, reconnect_interval: int = 5):
        super().__init__()
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer last 30 frames
        self.running = False
        self.connected = False
        self.last_frame = None
        self.fps = 0
        self._last_frame_time = time.time()
        self._frame_count = 0
        self._reconnect_interval = reconnect_interval
        self._last_reconnect_attempt = 0
        self._connection_errors = 0
        self._max_connection_errors = 5

    @contextmanager
    def timeout(self, seconds: int = 10):
        """Context manager for socket timeout."""
        old_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(seconds)
            yield
        finally:
            socket.setdefaulttimeout(old_timeout)

    def connect(self) -> bool:
        """Attempt to connect to the camera."""
        try:
            with self.timeout(10):  # 10 second timeout for connection
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened():
                    raise ConnectionError(f"Failed to open stream: {self.rtsp_url}")
                
                # Set OpenCV capture properties
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Minimize latency
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H265'))  # Set H.265 codec
                
                # Test frame grab with timeout
                if not cap.grab():
                    raise ConnectionError("Failed to grab test frame")
                
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    raise ConnectionError("Failed to retrieve test frame")
                
                # Verify frame dimensions
                height, width = frame.shape[:2]
                if width == 0 or height == 0:
                    raise ConnectionError("Invalid frame dimensions")
                
                self.connected = True
                self._connection_errors = 0
                return cap
            
        except Exception as e:
            self._connection_errors += 1
            print(f"Camera {self.camera_id} connection error: {str(e)}")
            if self._connection_errors >= self._max_connection_errors:
                print(f"Camera {self.camera_id} exceeded maximum connection attempts")
                self.running = False
            return None

    def run(self):
        """Main camera thread loop with automatic reconnection."""
        self.running = True
        cap = None
        
        while self.running:
            current_time = time.time()
            
            # Check if we need to reconnect
            if not self.connected and current_time - self._last_reconnect_attempt >= self._reconnect_interval:
                self._last_reconnect_attempt = current_time
                cap = self.connect()
                if cap is None:
                    time.sleep(1)
                    continue
            
            # Process frames
            try:
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        raise ConnectionError("Failed to read frame")
                    
                    # Calculate FPS
                    current_time = time.time()
                    self._frame_count += 1
                    if current_time - self._last_frame_time >= 1.0:
                        self.fps = self._frame_count
                        self._frame_count = 0
                        self._last_frame_time = current_time
                    
                    # Update frame buffer
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.last_frame = frame
                    self.frame_queue.put(frame)
                
            except Exception as e:
                print(f"Camera {self.camera_id} stream error: {str(e)}")
                self.connected = False
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(0.1)
        
        if cap is not None:
            cap.release()

    def stop(self):
        self.running = False
        self.join()

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame."""
        return self.last_frame

class CameraManager:
    def __init__(self, config_file: str = "camera_config.json"):
        self.config_file = config_file
        self.cameras: Dict[str, CameraStream] = {}
        self.camera_configs: Dict[str, CameraConfig] = {}
        self.nvr_settings = None
        self.stream_settings = None
        self.network_settings = None
        self.load_config()

    def load_config(self):
        """Load camera configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
                
                # Load NVR settings
                self.nvr_settings = config_data.get('nvr_settings', {})
                self.stream_settings = config_data.get('stream_settings', {})
                self.network_settings = config_data.get('network_settings', {})
                
                # Load camera configurations
                for camera in config_data['cameras']:
                    # Format RTSP URLs with credentials
                    rtsp_main = camera['rtsp_main'].format(
                        username=camera['username'],
                        password=camera['password'],
                        ip=camera['ip'],
                        port=camera['port']
                    )
                    rtsp_sub = camera.get('rtsp_sub', '').format(
                        username=camera['username'],
                        password=camera['password'],
                        ip=camera['ip'],
                        port=camera['port']
                    )
                    
                    config = CameraConfig(
                        camera_id=camera['id'],
                        rtsp_url=rtsp_main,  # Use main stream by default
                        zone=camera['zone'],
                        position=(camera['position']['x'], camera['position']['y'])
                    )
                    self.camera_configs[config.camera_id] = config
                    
        except FileNotFoundError:
            print(f"Config file {self.config_file} not found. Using default configuration.")
            self.create_default_config()
        except (KeyError, ValueError) as e:
            print(f"Error parsing camera configuration: {e}")
            self.create_default_config()

    def create_default_config(self):
        """Create a default configuration file."""
        default_config = {
            "nvr_settings": {
                "ip": "192.168.1.100",
                "port": 8000,
                "username": "admin",
                "password": "admin",
                "max_channels": 16
            },
            "cameras": [
                {
                    "id": "cam1",
                    "model": "AE-IPPR03D",
                    "ip": "192.168.1.101",
                    "port": 554,
                    "username": "admin",
                    "password": "admin",
                    "rtsp_main": "rtsp://{username}:{password}@{ip}:{port}/stream1",
                    "rtsp_sub": "rtsp://{username}:{password}@{ip}:{port}/stream2",
                    "zone": "entrance",
                    "position": {"x": 0, "y": 0},
                    "settings": {
                        "resolution": "2048x1536",
                        "fps": 20,
                        "ir_led": "auto"
                    }
                }
            ],
            "stream_settings": {
                "main_stream": {
                    "resolution": "2048x1536",
                    "fps": 20,
                    "encoding": "H.265"
                },
                "sub_stream": {
                    "resolution": "1920x1080",
                    "fps": 15,
                    "encoding": "H.265"
                }
            }
        }
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        # Load the default config
        self.load_config()

    def start_cameras(self):
        """Start all camera streams."""
        for camera_id, config in self.camera_configs.items():
            if camera_id not in self.cameras:
                try:
                    stream = CameraStream(camera_id, config.rtsp_url)
                    stream.start()
                    self.cameras[camera_id] = stream
                    print(f"Started camera {camera_id} successfully")
                except Exception as e:
                    print(f"Failed to start camera {camera_id}: {e}")

    def stop_cameras(self):
        """Stop all camera streams."""
        for camera in self.cameras.values():
            camera.stop()
        self.cameras.clear()

    def get_frames(self) -> Dict[str, np.ndarray]:
        """Get the most recent frame from each camera."""
        frames = {}
        for camera_id, stream in self.cameras.items():
            frame = stream.get_frame()
            if frame is not None:
                frames[camera_id] = frame
        return frames

    def get_camera_info(self, camera_id: str) -> Optional[CameraConfig]:
        """Get configuration information for a specific camera."""
        return self.camera_configs.get(camera_id)

    def validate_camera_config(self, camera_id: str, ip: str, username: str, password: str, 
                             zone: str, position: Tuple[int, int], model: str) -> Dict:
        """Validate camera configuration parameters."""
        # Validate camera ID
        if not camera_id or not isinstance(camera_id, str):
            raise ValueError("Invalid camera ID")
        
        # Validate IP address
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            raise ValueError(f"Invalid IP address: {ip}")
        
        # Validate credentials
        if not username or not password:
            raise ValueError("Username and password are required")
        
        # Validate model
        valid_models = ["AE-IPPR03D", "AE-IPPR03B"]
        if model not in valid_models:
            raise ValueError(f"Invalid camera model. Must be one of: {', '.join(valid_models)}")
        
        # Validate zone
        valid_zones = ["entrance", "checkout", "electronics", "clothing", "groceries", "home_goods"]
        if zone not in valid_zones:
            raise ValueError(f"Invalid zone. Must be one of: {', '.join(valid_zones)}")
        
        # Validate position
        if not isinstance(position, tuple) or len(position) != 2:
            raise ValueError("Position must be a tuple of (x, y) coordinates")
        if not all(isinstance(coord, int) and coord >= 0 for coord in position):
            raise ValueError("Position coordinates must be non-negative integers")
        
        # Create validated configuration
        return {
            "id": camera_id,
            "model": model,
            "ip": ip,
            "port": 554,  # Default RTSP port
            "username": username,
            "password": password,
            "rtsp_main": "rtsp://{username}:{password}@{ip}:{port}/stream1",
            "rtsp_sub": "rtsp://{username}:{password}@{ip}:{port}/stream2",
            "zone": zone,
            "position": {"x": position[0], "y": position[1]},
            "settings": {
                "resolution": "2048x1536",
                "fps": 20,
                "ir_led": "auto",
                "motion_detection": True,
                "detection_area": [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]
            }
        }

    def add_camera(self, camera_id: str, ip: str, username: str, password: str, 
                  zone: str, position: Tuple[int, int], model: str = "AE-IPPR03D"):
        """Add a new camera to the system."""
        try:
            # Check if camera ID already exists
            if camera_id in self.camera_configs:
                raise ValueError(f"Camera ID {camera_id} already exists")
            
            # Validate and create camera configuration
            camera_config = self.validate_camera_config(
                camera_id, ip, username, password, zone, position, model
            )
            
            # Format RTSP URL
            rtsp_url = camera_config['rtsp_main'].format(
                username=username,
                password=password,
                ip=ip,
                port=554
            )
            
            # Create CameraConfig object
            config = CameraConfig(camera_id, rtsp_url, zone, position)
            
            # Test camera connection before adding
            test_stream = CameraStream(camera_id, rtsp_url)
            test_stream.connect()
            if not test_stream.connected:
                raise ConnectionError(f"Failed to connect to camera at {ip}")
            test_stream.stop()
            
            # Update configuration
            self.camera_configs[camera_id] = config
            
            # Update config file
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            config_data['cameras'].append(camera_config)
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            # Start the camera stream
            stream = CameraStream(camera_id, rtsp_url)
            stream.start()
            self.cameras[camera_id] = stream
            print(f"Added and started camera {camera_id} successfully")
            
        except ValueError as e:
            print(f"Configuration error for camera {camera_id}: {str(e)}")
            raise
        except ConnectionError as e:
            print(f"Connection error for camera {camera_id}: {str(e)}")
            raise
        except Exception as e:
            print(f"Failed to add camera {camera_id}: {str(e)}")
            if camera_id in self.camera_configs:
                del self.camera_configs[camera_id]
            raise

    def remove_camera(self, camera_id: str):
        """Remove a camera from the system."""
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]
        
        if camera_id in self.camera_configs:
            del self.camera_configs[camera_id]
        
        # Update config file
        with open(self.config_file, 'r') as f:
            config_data = json.load(f)
        
        config_data['cameras'] = [
            cam for cam in config_data['cameras'] if cam['id'] != camera_id
        ]
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=4)

if __name__ == "__main__":
    # Example usage
    manager = CameraManager()
    manager.start_cameras()
    
    try:
        while True:
            frames = manager.get_frames()
            for camera_id, frame in frames.items():
                cv2.imshow(f"Camera {camera_id}", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        manager.stop_cameras()
        cv2.destroyAllWindows()
