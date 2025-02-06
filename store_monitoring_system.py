import cv2
import numpy as np
from camera_manager import CameraManager
from traffic_analytics import TrafficAnalytics
from FaceRecognizer import StoreVisitorRecognition
import threading
from datetime import datetime
import time
import os
from typing import Dict, List, Tuple
import json

class StoreMonitoringSystem:
    def __init__(self, store_dimensions: Tuple[int, int] = (800, 600)):
        # Initialize components
        self.camera_manager = CameraManager()
        self.face_recognizer = StoreVisitorRecognition()
        self.analytics = TrafficAnalytics(store_dimensions)
        
        # Display settings
        self.display_width = 1920
        self.display_height = 1080
        self.max_cameras_per_row = 3
        
        # Processing flags
        self.running = False
        self.save_interval = 300  # Save analytics every 5 minutes
        self.last_save_time = time.time()
        
        # Create output directory
        os.makedirs("output", exist_ok=True)

    def start(self):
        """Start the monitoring system."""
        self.running = True
        self.camera_manager.start_cameras()
        
        # Start processing thread
        processing_thread = threading.Thread(target=self._process_frames)
        processing_thread.start()
        
        try:
            self._display_feeds()
        finally:
            self.stop()
            processing_thread.join()

    def stop(self):
        """Stop the monitoring system."""
        self.running = False
        self.camera_manager.stop_cameras()
        cv2.destroyAllWindows()

    def _process_frames(self):
        """Process frames from all cameras in a separate thread."""
        while self.running:
            frames = self.camera_manager.get_frames()
            current_time = datetime.now()
            
            for camera_id, frame in frames.items():
                # Process frame with face recognition
                processed_frame = self.face_recognizer.process_frame(frame)
                
                # Extract face locations and update analytics
                faces = self.face_recognizer.app.get(frame)
                for face in faces:
                    bbox = face.bbox.astype(int)
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = (bbox[1] + bbox[3]) // 2
                    
                    # Get visitor ID from face recognizer
                    embedding = face.embedding
                    matching_id, confidence = self.face_recognizer.find_matching_face(embedding)
                    
                    if matching_id is None:
                        matching_id = self.face_recognizer.register_new_visitor(embedding, confidence)
                    
                    # Update analytics with visitor position
                    self.analytics.update_visitor_position(
                        matching_id,
                        (center_x, center_y),
                        camera_id,
                        current_time
                    )
            
            # Periodically save analytics
            if time.time() - self.last_save_time > self.save_interval:
                self.analytics.save_analytics_report()
                self.last_save_time = time.time()
            
            time.sleep(0.01)  # Small sleep to prevent CPU overload

    def _display_feeds(self):
        """Display all camera feeds in a grid layout."""
        while self.running:
            frames = self.camera_manager.get_frames()
            if not frames:
                time.sleep(0.1)
                continue
            
            # Calculate grid layout
            n_cameras = len(frames)
            grid_cols = min(n_cameras, self.max_cameras_per_row)
            grid_rows = (n_cameras + grid_cols - 1) // grid_cols
            
            # Calculate individual frame size
            frame_width = self.display_width // grid_cols
            frame_height = self.display_height // grid_rows
            
            # Create display grid
            display_grid = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
            
            # Place frames in grid
            for idx, (camera_id, frame) in enumerate(frames.items()):
                # Calculate position in grid
                grid_x = (idx % grid_cols) * frame_width
                grid_y = (idx // grid_cols) * frame_height
                
                # Resize frame to fit grid
                resized_frame = cv2.resize(frame, (frame_width, frame_height))
                
                # Add camera ID label
                cv2.putText(resized_frame, f"Camera {camera_id}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (255, 255, 255), 2)
                
                # Place in display grid
                display_grid[grid_y:grid_y+frame_height,
                           grid_x:grid_x+frame_width] = resized_frame
            
            # Show analytics overlay
            self._add_analytics_overlay(display_grid)
            
            # Display the grid
            cv2.imshow("Store Monitoring System", display_grid)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _add_analytics_overlay(self, display_grid):
        """Add real-time analytics overlay to the display."""
        # Get current statistics
        current_time = datetime.now()
        active_visitors = len(self.analytics.current_positions)
        total_visitors = len(self.analytics.visitor_metrics)
        avg_dwell_time = self.analytics.calculate_average_dwell_time()
        
        # Create overlay text
        overlay_text = [
            f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Active Visitors: {active_visitors}",
            f"Total Visitors Today: {total_visitors}",
            f"Avg Dwell Time: {avg_dwell_time.seconds // 60} mins"
        ]
        
        # Add text to display
        y_offset = 30
        for text in overlay_text:
            cv2.putText(display_grid, text,
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)
            y_offset += 30

    def add_camera(self, camera_id: str, rtsp_url: str, zone: str, position: Tuple[int, int]):
        """Add a new camera to the system."""
        self.camera_manager.add_camera(camera_id, rtsp_url, zone, position)

    def remove_camera(self, camera_id: str):
        """Remove a camera from the system."""
        self.camera_manager.remove_camera(camera_id)

if __name__ == "__main__":
    # Example usage
    system = StoreMonitoringSystem()
    
    # Add some test cameras (replace with actual RTSP URLs)
    system.add_camera("cam1", "0", "entrance", (0, 0))  # Use default webcam for testing
    
    try:
        system.start()
    except KeyboardInterrupt:
        system.stop()
