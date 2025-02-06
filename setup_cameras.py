#!/usr/bin/env python3
import json
import os
import ipaddress
import subprocess
import sys
from typing import Dict, List, Optional

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print setup header."""
    print("=" * 60)
    print("Store Traffic Monitoring System - Camera Setup")
    print("=" * 60)
    print()

def validate_ip(ip: str) -> bool:
    """Validate IP address format."""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False

def ping_ip(ip: str) -> bool:
    """Test if IP address is reachable."""
    param = '-n' if os.name == 'nt' else '-c'
    command = ['ping', param, '1', ip]
    return subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

def load_existing_config() -> Dict:
    """Load existing camera configuration if available."""
    try:
        with open('camera_config.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "nvr_settings": {
                "ip": "",
                "port": 8000,
                "username": "admin",
                "password": "admin",
                "max_channels": 16
            },
            "cameras": [],
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

def setup_nvr() -> Dict:
    """Configure NVR settings."""
    print("\nNVR Configuration")
    print("-" * 20)
    
    nvr_settings = {}
    while True:
        ip = input("Enter NVR IP address: ").strip()
        if validate_ip(ip):
            if ping_ip(ip):
                nvr_settings['ip'] = ip
                break
            else:
                print("Warning: IP address is not reachable. Do you want to continue? (y/n)")
                if input().lower() == 'y':
                    nvr_settings['ip'] = ip
                    break
        else:
            print("Invalid IP address format. Please try again.")
    
    nvr_settings['port'] = int(input("Enter NVR port (default 8000): ").strip() or "8000")
    nvr_settings['username'] = input("Enter NVR username (default admin): ").strip() or "admin"
    nvr_settings['password'] = input("Enter NVR password (default admin): ").strip() or "admin"
    nvr_settings['max_channels'] = 16
    
    return nvr_settings

def setup_camera() -> Dict:
    """Configure a single camera."""
    camera = {}
    
    print("\nCamera Configuration")
    print("-" * 20)
    
    # Basic settings
    camera['id'] = input("Enter camera ID (e.g., cam1): ").strip()
    camera['model'] = input("Enter camera model (AE-IPPR03D/AE-IPPR03B): ").strip() or "AE-IPPR03D"
    
    while True:
        ip = input("Enter camera IP address: ").strip()
        if validate_ip(ip):
            if ping_ip(ip):
                camera['ip'] = ip
                break
            else:
                print("Warning: IP address is not reachable. Do you want to continue? (y/n)")
                if input().lower() == 'y':
                    camera['ip'] = ip
                    break
        else:
            print("Invalid IP address format. Please try again.")
    
    camera['port'] = int(input("Enter RTSP port (default 554): ").strip() or "554")
    camera['username'] = input("Enter camera username (default admin): ").strip() or "admin"
    camera['password'] = input("Enter camera password (default admin): ").strip() or "admin"
    
    # Zone settings
    print("\nAvailable zones: entrance, checkout, electronics, clothing, groceries, home_goods")
    camera['zone'] = input("Enter camera zone: ").strip().lower()
    
    # Position settings
    print("\nEnter camera position (x,y coordinates in store layout)")
    camera['position'] = {
        'x': int(input("X coordinate: ").strip() or "0"),
        'y': int(input("Y coordinate: ").strip() or "0")
    }
    
    # Stream URLs
    camera['rtsp_main'] = "rtsp://{username}:{password}@{ip}:{port}/stream1"
    camera['rtsp_sub'] = "rtsp://{username}:{password}@{ip}:{port}/stream2"
    
    # Camera settings
    camera['settings'] = {
        "resolution": "2048x1536",
        "fps": 20,
        "ir_led": "auto",
        "motion_detection": True,
        "detection_area": [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]
    }
    
    return camera

def save_config(config: Dict):
    """Save configuration to file."""
    with open('camera_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    print("\nConfiguration saved to camera_config.json")

def test_camera(camera: Dict):
    """Test camera connection using test_camera_connection.py."""
    print(f"\nTesting camera {camera['id']}...")
    cmd = [
        sys.executable, 'test_camera_connection.py',
        '--camera-id', camera['id'],
        '--ip', camera['ip'],
        '--username', camera['username'],
        '--password', camera['password'],
        '--display',
        '--duration', '5'
    ]
    subprocess.run(cmd)

def main():
    clear_screen()
    print_header()
    
    # Load existing config or create new
    config = load_existing_config()
    
    # Setup NVR
    print("Do you want to configure NVR settings? (y/n)")
    if input().lower() == 'y':
        config['nvr_settings'] = setup_nvr()
    
    while True:
        clear_screen()
        print_header()
        
        print("\nCurrent Configuration:")
        print(f"NVR IP: {config['nvr_settings']['ip']}")
        print(f"Number of cameras: {len(config['cameras'])}")
        
        print("\nOptions:")
        print("1. Add new camera")
        print("2. Test all cameras")
        print("3. Save and exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            camera = setup_camera()
            config['cameras'].append(camera)
            save_config(config)
            
            print("\nDo you want to test this camera now? (y/n)")
            if input().lower() == 'y':
                test_camera(camera)
        
        elif choice == '2':
            subprocess.run([sys.executable, 'test_camera_connection.py', '--all'])
            input("\nPress Enter to continue...")
        
        elif choice == '3':
            save_config(config)
            print("\nSetup complete! You can now run the monitoring system:")
            print("python store_monitoring_system.py")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        sys.exit(0)
