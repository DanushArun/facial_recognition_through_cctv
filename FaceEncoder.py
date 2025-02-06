import os
import csv
import face_recognition
import cv2

class FaceEncoder:
    def __init__(self):
        self.input_dir = "Augmented_Images"
        self.output_csv = "EncodedFaces.csv"
        
    def encode_faces(self):
        """Encode faces from images and save to CSV"""
        with open(self.output_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            # CSV header
            writer.writerow(["Name", "Encoding"])  
            
            for filename in os.listdir(self.input_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name = os.path.splitext(filename)[0]
                    file_path = os.path.join(self.input_dir, filename)
                    
                    # Load image
                    image = face_recognition.load_image_file(file_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if len(encodings) > 0:
                        # We'll store only the first encoding for simplicity
                        encoding_list = encodings[0].tolist()
                        writer.writerow([name, encoding_list])
                        print(f"Encoded {filename}")
                    else:
                        print(f"No face found in {filename}")

if __name__ == "__main__":
    encoder = FaceEncoder()
    encoder.encode_faces()
