import cv2
import csv
import os
import numpy as np
from datetime import datetime
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import faiss
import pickle

CSV_FILENAME = "Visitors.csv"
EMBEDDINGS_FILE = "face_embeddings.pkl"

class StoreVisitorRecognition:
    def __init__(self):
        # Initialize InsightFace
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize FAISS index (L2 normalized for cosine similarity)
        self.dimension = 512  # InsightFace embedding dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product index for cosine similarity
        
        # Lists to keep track of known faces
        self.known_ids = []
        self.visitor_data = {}
        self.face_embeddings = []  # Store embeddings for persistence
        
        # ID auto-increment
        self.next_id = 1
        
        # Frame skipping to improve performance
        self.frame_skip = 2
        self.frame_count = 0
        
        # Face recognition threshold (lower value = more strict matching)
        self.recognition_threshold = 0.6  # Increased threshold for more strict matching
        
        # Load existing data from CSV (if any)
        self.load_csv()
    
    def load_csv(self):
        """Load existing visitors from CSV and embeddings from pickle file."""
        if not os.path.exists(CSV_FILENAME):
            # Create a new CSV with headers if none exists
            with open(CSV_FILENAME, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Visitor_Number", "ID", "Login_Time", "Confidence_Score"])
            return
        
        # Load face embeddings if they exist
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, 'rb') as f:
                self.face_embeddings = pickle.load(f)
                # Rebuild FAISS index
                if self.face_embeddings:
                    embeddings_array = np.array(self.face_embeddings)
                    faiss.normalize_L2(embeddings_array)
                    self.index.reset()
                    self.index.add(embeddings_array)
        
        with open(CSV_FILENAME, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                id_str = row["ID"]
                self.known_ids.append(id_str)
                
                # Store visitor data
                visitor_number = row["Visitor_Number"]
                login_time = row["Login_Time"]
                confidence = row["Confidence_Score"]
                
                # Update next_id so we don't reuse IDs
                numeric_id = int(id_str.replace("Ranger_", "")) if "Ranger_" in id_str else 0
                self.next_id = max(self.next_id, numeric_id + 1)
                
                self.visitor_data[id_str] = {
                    "visitor_number": visitor_number,
                    "login_time": login_time,
                    "confidence": confidence
                }
    
    def save_csv(self):
        """Overwrite the CSV with current known data."""
        with open(CSV_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["Visitor_Number", "ID", "Login_Time", "Confidence_Score"])
            
            # Each row
            for idx, id_str in enumerate(self.known_ids):
                visitor_number = str(idx + 1)  # 1-based visitor numbering
                login_time = self.visitor_data[id_str]["login_time"]
                confidence = self.visitor_data[id_str]["confidence"]
                
                writer.writerow([
                    visitor_number,
                    id_str,
                    login_time,
                    confidence
                ])
    
    def register_new_visitor(self, face_embedding, confidence):
        """Create a new visitor ID, store face embedding and details, return the ID."""
        new_id = f"Ranger_{self.next_id}"
        self.next_id += 1
        
        # Fill data
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Normalize embedding for FAISS
        normalized_embedding = face_embedding.reshape(1, -1).copy()
        faiss.normalize_L2(normalized_embedding)
        
        # Add to storage
        self.face_embeddings.append(normalized_embedding.flatten())
        self.known_ids.append(new_id)
        
        # Add to FAISS index
        self.index.add(normalized_embedding)
        
        self.visitor_data[new_id] = {
            "visitor_number": str(len(self.known_ids)),  # Current count becomes visitor number
            "login_time": now_str,
            "confidence": f"{confidence:.2f}"
        }
        
        # Persist data
        self.save_csv()
        self.save_embeddings()
        
        return new_id
    
    def save_embeddings(self):
        """Save face embeddings to pickle file."""
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(self.face_embeddings, f)
    
    def find_matching_face(self, face_embedding):
        """Find the closest matching face using FAISS."""
        if self.index.ntotal == 0:
            return None, 0
        
        # Normalize query embedding
        query_embedding = face_embedding.reshape(1, -1).copy()
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, 1)
        
        # FAISS returns similarities in range [-1, 1], convert to [0, 1]
        similarity = float(distances[0][0] + 1) / 2
        
        # Check if the similarity is above threshold
        if similarity > self.recognition_threshold:
            return self.known_ids[indices[0][0]], similarity * 100
        
        return None, 0
    
    def process_frame(self, frame):
        """Detect and recognize faces using InsightFace."""
        self.frame_count += 1
        
        # Skip frames to speed up
        if self.frame_count % self.frame_skip != 0:
            return frame
        
        # Detect faces
        faces = self.app.get(frame)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            
            # Try to find a match
            matching_id, confidence = self.find_matching_face(embedding)
            
            if matching_id is None:
                # New face
                matching_id = self.register_new_visitor(embedding, confidence)
                color = (0, 255, 0)  # Green for new
            else:
                color = (0, 0, 255)  # Red for existing
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label_text = f"{matching_id} | {confidence:.1f}%"
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
        
        return frame
    
    def run_camera(self, source=0):
        """Start webcam or camera feed and process frames."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open camera/video source.")
            return
        
        print("Press 'q' to quit...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            cv2.imshow("Store Visitor Recognition", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    svc = StoreVisitorRecognition()
    svc.run_camera()
