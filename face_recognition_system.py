import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import supervision as sv
import pandas as pd
from datetime import datetime
import os
import face_recognition
import time

class FaceRecognitionSystem:
    def __init__(self):
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Initialize known face encodings
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
        # Initialize attendance tracking
        self.attendance = {}
        self.person_ids = {}  # To store unique IDs for each person
        self.next_id = 1      # Counter for generating unique IDs
        
        # Performance optimization settings
        self.frame_skip = 2   # Process every nth frame
        self.frame_count = 0
        self.min_face_size = 30       # Minimum face size to detect
        self.confidence_threshold = 0.6
        
        # Check for GPU availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create attendance CSV with headers if it doesn't exist
        if not os.path.exists('Attendance.csv'):
            df = pd.DataFrame(columns=['ID', 'Name', 'In Time', 'Out Time'])
            df.to_csv('Attendance.csv', index=False)
        
    def load_known_faces(self):
        """Load and encode faces from the Original_Images directory"""
        image_dir = "Original_Images"
        if not os.path.exists(image_dir):
            print(f"Directory '{image_dir}' does not exist. Skipping known face loading.")
            return

        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                # Extract name from filename (assuming the filename is the person's name)
                name = os.path.splitext(filename)[0]
                
                # Load and encode face
                image_path = os.path.join(image_dir, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(name)
                    print(f"Loaded known face for {name}")

    def recognize_face(self, face_image):
        """Compare a face image with known faces and return the name if recognized"""
        face_encodings = face_recognition.face_encodings(face_image)
        if not face_encodings:
            return None
        
        face_encoding = face_encodings[0]
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
        
        if True in matches:
            match_index = matches.index(True)
            return self.known_face_names[match_index]
        
        return None
    
    def mark_attendance(self, name):
        """Mark attendance for a recognized person"""
        current_time = datetime.now()
        
        # Generate unique ID for new person
        if name not in self.person_ids:
            self.person_ids[name] = self.next_id
            self.next_id += 1
        
        person_id = self.person_ids[name]
        
        # Handle attendance marking
        if name not in self.attendance:
            # First time entry - mark in time
            self.attendance[name] = {
                'id': person_id,
                'in_time': current_time,
                'out_time': None
            }
        else:
            # Update out time if more than 1 minute has passed since in_time
            if (current_time - self.attendance[name]['in_time']).total_seconds() > 60:
                self.attendance[name]['out_time'] = current_time
        
        # Save to CSV
        df = pd.DataFrame(columns=['ID', 'Name', 'In Time', 'Out Time'])
        for person, data in self.attendance.items():
            df = df._append({
                'ID': data['id'],
                'Name': person,
                'In Time': data['in_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'Out Time': data['out_time'].strftime('%Y-%m-%d %H:%M:%S') if data['out_time'] else 'Still Present'
            }, ignore_index=True)
        df.to_csv('Attendance.csv', index=False)

    def process_frame(self, frame):
        """Process a single frame for face detection and recognition,
           drawing bounding boxes for both recognized and unrecognized faces."""
        # Skip frames for better performance
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return frame

        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_frame.shape
        
        # Try face_recognition's face detection first
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        # If no faces found with face_recognition, try YOLO for "person" detection
        if not face_locations:
            results = self.model(frame, conf=self.confidence_threshold, classes=[0])  # class 0 => person
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get coordinates of person
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Calculate person height
                    person_height = y2 - y1
                    
                    # Extract approximate face region (upper portion of the bounding box)
                    face_top = y1
                    face_bottom = y1 + int(person_height * 0.3)
                    face_left = x1 + int((x2 - x1) * 0.25)
                    face_right = x2 - int((x2 - x1) * 0.25)
                    
                    # Clip coords to image boundaries
                    face_top = max(face_top, 0)
                    face_left = max(face_left, 0)
                    face_bottom = min(face_bottom, height)
                    face_right = min(face_right, width)
                    
                    # Filter invalid or tiny face regions
                    if face_bottom <= face_top or face_right <= face_left:
                        continue
                    
                    face_locations.append((face_top, face_right, face_bottom, face_left))
        
        # Process each detected face
        for face_location in face_locations:
            top, right, bottom, left = face_location
            
            # Clip bounding box to avoid negative or out-of-bounds
            top = max(top, 0)
            left = max(left, 0)
            bottom = min(bottom, height)
            right = min(right, width)
            
            if bottom <= top or right <= left:
                continue
            
            # Filter out very small faces
            face_height = bottom - top
            if face_height < self.min_face_size:
                continue
            
            # Default bounding box color (red) for unknown
            box_color = (0, 0, 255)
            label_text = "Unknown"
            
            # Extract face region
            face_image = rgb_frame[top:bottom, left:right]
            
            if face_image.size != 0:
                # Attempt to recognize face
                name = self.recognize_face(face_image)
                
                if name:
                    # Mark attendance
                    self.mark_attendance(name)
                    
                    # Use a different color (green) for recognized faces
                    # and label "Re-visiting: {name}"
                    box_color = (0, 255, 0)
                    label_text = f"Re-visiting: {name}"
            
            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(frame,
                          (left, top - label_height - 10),
                          (left + label_width, top),
                          box_color,
                          -1)
            
            # Draw label text in black
            cv2.putText(frame, label_text,
                        (left, top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 2)
        
        return frame
    
    def start_recognition(self, source=0):
        """Start face recognition from video source"""
        max_attempts = 3
        attempt = 0
        cap = None
        
        # Set buffer size for better performance
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtbufsize=1024M'
        
        # Attempt to open camera
        while attempt < max_attempts:
            try:
                cap = cv2.VideoCapture(source)
                if cap is not None and cap.isOpened():
                    break
            except Exception as e:
                print(f"Attempt {attempt + 1}: Failed to open camera - {str(e)}")
            
            attempt += 1
            if attempt < max_attempts:
                print("Retrying camera initialization...")
                time.sleep(1)
        
        if cap is None or not cap.isOpened():
            print("Error: Could not open camera after multiple attempts")
            return
            
        print("Camera started successfully. Press 'q' to quit.")
        
        # Set camera properties (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffer
        
        frame_count = 0
        error_count = 0
        max_errors = 5
        
        try:
            while True:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        error_count += 1
                        print(f"Error reading frame ({error_count}/{max_errors})")
                        if error_count >= max_errors:
                            print("Too many consecutive errors, stopping...")
                            break
                        continue
                    
                    error_count = 0  # Reset error count on successful frame read
                    frame_count += 1
                    
                    # Process frame (catch any face_recognition/dlib errors)
                    try:
                        processed_frame = self.process_frame(frame)
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        error_count += 1
                        if error_count >= max_errors:
                            print("Too many consecutive errors, stopping...")
                            break
                        continue
                    
                    # Display result
                    cv2.imshow('Face Recognition Attendance System', processed_frame)
                    
                    # Break loop on 'q' press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quit command received")
                        # Update final out times for everyone still present
                        current_time = datetime.now()
                        for name, data in self.attendance.items():
                            if data['out_time'] is None:
                                data['out_time'] = current_time
                        break
                        
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    error_count += 1
                    if error_count >= max_errors:
                        print("Too many consecutive errors, stopping...")
                        break
                    continue
                    
        finally:
            # Save final attendance state
            df = pd.DataFrame(columns=['ID', 'Name', 'In Time', 'Out Time'])
            for person, data in self.attendance.items():
                df = df._append({
                    'ID': data['id'],
                    'Name': person,
                    'In Time': data['in_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Out Time': data['out_time'].strftime('%Y-%m-%d %H:%M:%S') if data['out_time'] else 'Still Present'
                }, ignore_index=True)
            df.to_csv('Attendance.csv', index=False)
            
            cap.release()
            cv2.destroyAllWindows()
            print("\nAttendance has been saved to Attendance.csv")


if __name__ == "__main__":
    # Initialize and start face recognition system
    face_system = FaceRecognitionSystem()
    face_system.start_recognition()
