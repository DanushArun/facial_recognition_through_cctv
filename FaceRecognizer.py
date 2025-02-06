import cv2
import csv
import os
import time
import numpy as np
from datetime import datetime
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import faiss
import pickle
from PIL import Image
import pandas as pd

CSV_FILENAME = "Visitors.csv"
EMBEDDINGS_FILE = "face_embeddings.pkl"

class StoreVisitorRecognition:
    def __init__(self):
        # Initialize InsightFace with optimized settings for CCTV
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(1280, 1280))  # Much larger detection size for CCTV
        print("Face detection model initialized with size (1280, 1280)")
        
        # Create Visitors.csv if it doesn't exist
        if not os.path.exists(CSV_FILENAME):
            with open(CSV_FILENAME, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "First_Entry_Time", "Last_Exit_Time", "Visit_Count", "Confidence"])
        
        # Initialize FAISS index (L2 normalized for cosine similarity)
        self.dimension = 512  # InsightFace embedding dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Lists to keep track of known faces
        self.known_ids = []
        self.visitor_data = {}
        self.face_embeddings = []
        
        self.next_id = 1
        self.frame_skip = 1  # Process every frame for better detection
        self.frame_count = 0
        
        # Optimized face recognition parameters for CCTV
        self.recognition_threshold = 0.35  # Balanced threshold
        self.min_face_quality = 0.45  # Higher quality threshold for better accuracy
        self.max_angle = 45  # More reasonable angle tolerance
        self.min_face_size = 60  # Larger minimum face size for better recognition
        
        # Optimized temporal matching
        self.recent_matches = {}
        self.match_history_size = 3  # Smaller history for faster matching
        self.temporal_consistency_threshold = 0.35  # Lower consistency requirement
        
        # Minimal preprocessing requirements
        self.target_face_size = (112, 112)  # Standard size for recognition
        self.brightness_threshold = 20  # Much lower brightness requirement
        self.contrast_threshold = 15  # Much lower contrast requirement
        
        # Remove old embeddings file if it exists
        if os.path.exists(EMBEDDINGS_FILE):
            os.remove(EMBEDDINGS_FILE)
        
        self.load_csv()
    
    def update_visitor_record(self, visitor_id, confidence, is_exit=False):
        """Update visitor record in CSV"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            df = pd.read_csv(CSV_FILENAME)
            visitor_exists = df['ID'] == visitor_id
            
            if any(visitor_exists):
                # Update existing visitor
                idx = visitor_exists.idxmax()
                if is_exit:
                    df.at[idx, 'Last_Exit_Time'] = current_time
                df.at[idx, 'Visit_Count'] = df.at[idx, 'Visit_Count'] + 1
                df.at[idx, 'Confidence'] = f"{confidence:.2f}"
            else:
                # Add new visitor
                new_record = pd.DataFrame({
                    'ID': [visitor_id],
                    'First_Entry_Time': [current_time],
                    'Last_Exit_Time': [''],
                    'Visit_Count': [1],
                    'Confidence': [f"{confidence:.2f}"]
                })
                df = pd.concat([df, new_record], ignore_index=True)
            
            df.to_csv(CSV_FILENAME, index=False)
            return not any(visitor_exists)  # Return True if new visitor
            
        except Exception as e:
            print(f"Error updating visitor record: {str(e)}")
            return False

    def preprocess_frame(self, frame):
        """Enhanced frame preprocessing for CCTV footage"""
        try:
            # Maintain higher resolution for CCTV
            height, width = frame.shape[:2]
            target_size = 1920  # Higher resolution for CCTV
            scale = target_size / max(height, width)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
            # Denoise first
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
            
            # Enhance contrast with CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Sharpen
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Adjust brightness and contrast
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=5)
            
            return enhanced
            
        except Exception as e:
            print(f"Frame preprocessing error: {str(e)}")
            return frame

    def preprocess_face(self, frame, face):
        """Enhanced face preprocessing"""
        try:
            bbox = face.bbox.astype(int)
            landmarks = face.landmark_2d_106 if face.landmark_2d_106 is not None else face.landmark_3d_68
            
            if landmarks is None:
                return None
            
            # Extract and validate face region
            x1, y1, x2, y2 = bbox
            face_img = frame[int(y1):int(y2), int(x1):int(x2)]
            
            if face_img.shape[0] < self.min_face_size or face_img.shape[1] < self.min_face_size:
                return None
            
            # Enhanced quality checks
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            if brightness < self.brightness_threshold or contrast < self.contrast_threshold:
                return None
            
            # Improved face alignment
            src = np.array([
                [landmarks[38]], # Left eye
                [landmarks[88]], # Right eye
                [landmarks[33]], # Nose
                [landmarks[62]], # Mouth
            ], dtype=np.float32)
            
            dst = np.array([
                [0.3 * self.target_face_size[0], 0.3 * self.target_face_size[1]],
                [0.7 * self.target_face_size[0], 0.3 * self.target_face_size[1]],
                [0.5 * self.target_face_size[0], 0.6 * self.target_face_size[1]],
                [0.5 * self.target_face_size[0], 0.8 * self.target_face_size[1]]
            ], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(src, dst)
            aligned_face = cv2.warpPerspective(face_img, M, self.target_face_size)
            
            # Additional enhancements
            aligned_face = cv2.detailEnhance(aligned_face, sigma_s=10, sigma_r=0.15)
            
            return aligned_face
            
        except Exception as e:
            print(f"Face preprocessing error: {str(e)}")
            return None

    def process_frame(self, frame):
        """Enhanced frame processing with attendance tracking and debug info"""
        self.frame_count += 1
        
        if self.frame_count % self.frame_skip != 0:
            return frame
        
        # Preprocess frame
        enhanced_frame = self.preprocess_frame(frame)
        
        try:
            # Get faces with debug info
            faces = self.app.get(enhanced_frame)
            if len(faces) > 0:
                print(f"Detected {len(faces)} faces in frame")
            
            # Draw detection boxes with confidence scores
            for face in faces:
                bbox = face.bbox.astype(int)
                # Draw yellow box with detection score
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1)
                score_text = f"Score: {face.det_score:.2f}"
                cv2.putText(frame, score_text,
                          (bbox[0], bbox[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (0, 255, 255), 1)
            
            # Process valid faces
            for face in faces:
                if not self.is_face_valid(face, enhanced_frame):
                    print(f"Face invalid - Quality: {face.det_score:.2f}, Size: {face.bbox[2]-face.bbox[0]:.0f}x{face.bbox[3]-face.bbox[1]:.0f}")
                    continue
                    
                print(f"Processing valid face - Quality: {face.det_score:.2f}")
                
                processed_face = self.preprocess_face(enhanced_frame, face)
                if processed_face is None:
                    continue
                
                bbox = face.bbox.astype(int)
                embedding = face.embedding
            
                # Enhanced matching with temporal consistency
                temporal_match = self.check_temporal_consistency(embedding)
                if temporal_match:
                    matching_id, confidence = temporal_match
                else:
                    matching_id, confidence = self.find_matching_face(embedding)
                
                if matching_id is None:
                    matching_id = self.register_new_visitor(embedding, confidence)
                    # Green box for new visitors
                    color = (0, 255, 0)
                    is_new = self.update_visitor_record(matching_id, confidence)
                    print(f"New visitor detected: {matching_id}")
                else:
                    # Red box for returning visitors
                    color = (0, 0, 255)
                    self.update_visitor_record(matching_id, confidence)
                    print(f"Returning visitor: {matching_id}")
                
                # Enhanced visualization
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Add visit count to label
                df = pd.read_csv(CSV_FILENAME)
                visitor_data = df[df['ID'] == matching_id].iloc[0]
                visit_count = visitor_data['Visit_Count']
                
                label_text = f"{matching_id} | Visits: {visit_count} | {confidence:.1f}%"
                (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame,
                            (bbox[0], bbox[1] - label_height - 10),
                            (bbox[0] + label_width, bbox[1]),
                            color,
                            -1)
                
                cv2.putText(frame, label_text,
                            (bbox[0], bbox[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            
        return frame

    # [Previous methods remain unchanged: load_csv, save_csv, register_new_visitor, 
    # save_embeddings, check_temporal_consistency, find_matching_face, is_face_valid, run_cameras]
    
    def load_csv(self):
        """Load existing visitor data and embeddings"""
        if not os.path.exists(CSV_FILENAME):
            with open(CSV_FILENAME, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "First_Entry_Time", "Last_Exit_Time", "Visit_Count", "Confidence"])
            return
        
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, 'rb') as f:
                self.face_embeddings = pickle.load(f)
                if self.face_embeddings:
                    embeddings_array = np.array(self.face_embeddings)
                    faiss.normalize_L2(embeddings_array)
                    self.index.reset()
                    self.index.add(embeddings_array)
        
        try:
            df = pd.read_csv(CSV_FILENAME)
            for _, row in df.iterrows():
                id_str = row["ID"]
                self.known_ids.append(id_str)
                
                numeric_id = int(id_str.replace("Ranger_", "")) if "Ranger_" in id_str else 0
                self.next_id = max(self.next_id, numeric_id + 1)
                
                self.visitor_data[id_str] = {
                    "first_entry": row["First_Entry_Time"],
                    "last_exit": row["Last_Exit_Time"],
                    "visit_count": row["Visit_Count"],
                    "confidence": row["Confidence"]
                }
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
    
    def save_csv(self):
        """Save visitor data to CSV"""
        try:
            df = pd.DataFrame([
                {
                    "ID": id_str,
                    "First_Entry_Time": self.visitor_data[id_str]["first_entry"],
                    "Last_Exit_Time": self.visitor_data[id_str]["last_exit"],
                    "Visit_Count": self.visitor_data[id_str]["visit_count"],
                    "Confidence": self.visitor_data[id_str]["confidence"]
                }
                for id_str in self.known_ids
            ])
            df.to_csv(CSV_FILENAME, index=False)
        except Exception as e:
            print(f"Error saving CSV: {str(e)}")
    
    def register_new_visitor(self, face_embedding, confidence):
        """Register a new visitor"""
        new_id = f"Ranger_{self.next_id}"
        self.next_id += 1
        
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        normalized_embedding = face_embedding.reshape(1, -1).copy()
        faiss.normalize_L2(normalized_embedding)
        
        self.face_embeddings.append(normalized_embedding.flatten())
        self.known_ids.append(new_id)
        
        self.index.add(normalized_embedding)
        
        self.recent_matches[new_id] = [normalized_embedding.copy()]
        
        self.visitor_data[new_id] = {
            "first_entry": now_str,
            "last_exit": "",
            "visit_count": 1,
            "confidence": f"{confidence:.2f}"
        }
        
        self.save_csv()
        self.save_embeddings()
        
        return new_id
    
    def save_embeddings(self):
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(self.face_embeddings, f)
    
    def check_temporal_consistency(self, current_embedding):
        if not self.recent_matches:
            return None
            
        for visitor_id, recent_embeddings in self.recent_matches.items():
            similarities = [np.dot(current_embedding.flatten(), recent.flatten())
                          for recent in recent_embeddings]
            avg_similarity = np.mean(similarities)
            
            if avg_similarity > self.temporal_consistency_threshold:
                return visitor_id, avg_similarity * 100
        
        return None
    
    def find_matching_face(self, face_embedding):
        if self.index.ntotal == 0:
            print("Warning: FAISS index is empty. No known faces to compare against.")
            return None, 0
        
        if face_embedding is None or not isinstance(face_embedding, np.ndarray):
            print("Error: Invalid face embedding provided")
            return None, 0
            
        try:
            query_embedding = face_embedding.reshape(1, -1).copy()
            faiss.normalize_L2(query_embedding)
            
            distances, indices = self.index.search(query_embedding, min(3, self.index.ntotal))
            
            if (len(distances) == 0 or len(indices) == 0 or 
                len(distances[0]) == 0 or len(indices[0]) == 0):
                print("Warning: No matching faces found in the index")
                return None, 0
            
            for i in range(len(indices[0])):
                best_index = indices[0][i]
                
                if best_index >= len(self.known_ids):
                    continue
                
                visitor_id = self.known_ids[best_index]
                
                similarity = float(distances[0][i] + 1) / 2
                
                if similarity > self.recognition_threshold:
                    if visitor_id not in self.recent_matches:
                        self.recent_matches[visitor_id] = []
                    self.recent_matches[visitor_id].append(query_embedding.copy())
                    
                    if len(self.recent_matches[visitor_id]) > self.match_history_size:
                        self.recent_matches[visitor_id].pop(0)
                    
                    return visitor_id, similarity * 100
            
            print(f"No matches above threshold {self.recognition_threshold}")
            return None, 0
                
        except Exception as e:
            print(f"Face matching error: {str(e)}")
            return None, 0
    
    def is_face_valid(self, face, frame):
        """Check if face meets quality criteria with detailed logging"""
        # Log all validation steps
        print(f"\nFace Validation Details:")
        print(f"Detection Score: {face.det_score:.3f} (min: {self.min_face_quality})")
        
        if face.det_score < self.min_face_quality:
            print("Failed: Detection score too low")
            return False
            
        landmarks = face.landmark_3d_68 if face.landmark_3d_68 is not None else face.landmark_2d_106
        if landmarks is None:
            print("Failed: No landmarks detected")
            return False
            
        # Calculate face angles
        left_eye = landmarks[36]
        right_eye = landmarks[45]
        nose_tip = landmarks[33]
        
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        roll_angle = abs(np.degrees(np.arctan2(dy, dx)))
        
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        yaw_angle = abs(np.degrees(np.arctan2(nose_tip[0] - eye_center_x, nose_tip[2])))
        
        print(f"Roll Angle: {roll_angle:.1f}° (max: {self.max_angle}°)")
        print(f"Yaw Angle: {yaw_angle:.1f}° (max: {self.max_angle}°)")
        
        if roll_angle > self.max_angle or yaw_angle > self.max_angle:
            print("Failed: Face angles too extreme")
            return False
        
        # Get face size
        bbox = face.bbox.astype(int)
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        print(f"Face Size: {face_width}x{face_height} (min: {self.min_face_size})")
        
        if face_width < self.min_face_size or face_height < self.min_face_size:
            print("Failed: Face too small")
            return False
        
        processed_face = self.preprocess_face(frame, face)
        if processed_face is None:
            print("Failed: Face preprocessing failed")
            return False
        
        print("Success: Face passed all validation checks")
        return True

    def run_cameras(self):
        from camera_manager import CameraManager
        
        print("Initializing camera system...")
        
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("Using local webcam for testing...")
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            print("Error reading from webcam")
                            break
                        
                        processed_frame = self.process_frame(frame)
                        
                        cv2.imshow("Face Recognition Test (Webcam)", processed_frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Switching to RTSP cameras...")
                            break
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
            else:
                print("No local webcam available, trying RTSP cameras...")
        except Exception as e:
            print(f"Webcam error: {str(e)}, trying RTSP cameras...")
        
        print("\nInitializing RTSP cameras...")
        manager = CameraManager()
        manager.start_cameras()
        
        print("Processing camera feeds. Press 'q' to quit...")
        
        try:
            retry_count = 0
            max_retries = 5
            
            while True:
                frames = manager.get_frames()
                
                if not frames:
                    retry_count += 1
                    print(f"No camera feeds available. Retry {retry_count}/{max_retries}")
                    if retry_count >= max_retries:
                        print("Failed to get camera feeds after multiple attempts.")
                        print("Please check camera connections and configuration.")
                        break
                    time.sleep(2)
                    continue
                else:
                    retry_count = 0
                
                for camera_id, frame in frames.items():
                    camera_info = manager.get_camera_info(camera_id)
                    zone = camera_info.zone if camera_info else "unknown"
                    
                    processed_frame = self.process_frame(frame)
                    
                    cv2.putText(processed_frame, 
                              f"Camera: {camera_id} | Zone: {zone}",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                              (255, 255, 255), 2)
                    
                    window_name = f"Camera {camera_id} - {zone}"
                    cv2.imshow(window_name, processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Error processing camera feeds: {str(e)}")
        finally:
            manager.stop_cameras()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    svc = StoreVisitorRecognition()
