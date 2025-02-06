import cv2
import os

class ImagePreprocessor:
    def __init__(self):
        self.input_dir = "Temp_Images"
        self.output_dir = "Original_Images"
        self.target_size = (512, 512)  # or your choice
        
    def crop_to_square(self, image):
        """Crop image to square shape taking the center portion"""
        height, width = image.shape[:2]
        min_dim = min(height, width)
        
        # Calculate cropping boundaries
        start_y = (height - min_dim) // 2
        start_x = (width - min_dim) // 2
        
        # Crop the image
        cropped = image[start_y:start_y + min_dim, start_x:start_x + min_dim]
        return cropped
    
    def preprocess_image(self, filename):
        """Preprocess a single image"""
        # Read image
        image_path = os.path.join(self.input_dir, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error loading image: {filename}")
            return False
        
        try:
            # Crop to square
            squared_image = self.crop_to_square(image)
            
            # Resize
            resized_image = cv2.resize(squared_image, self.target_size)
            
            # Save preprocessed image
            output_path = os.path.join(self.output_dir, filename)
            cv2.imwrite(output_path, resized_image)
            
            return True
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return False
    
    def process_all_images(self):
        """Process all images in the input directory"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        success_count = 0
        total_count = 0
        
        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_count += 1
                if self.preprocess_image(filename):
                    success_count += 1
                    
        print(f"Successfully processed {success_count} out of {total_count} images")

if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    preprocessor.process_all_images()
