import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance

class DataAugmentor:
    def __init__(self):
        self.input_dir = "Original_Images"
        self.output_dir = "Augmented_Images"
        
    def rotate_image(self, image, angle):
        """Rotate image by given angle"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))
    
    def horizontal_flip(self, image):
        """Flip image horizontally"""
        return cv2.flip(image, 1)
    
    def adjust_brightness(self, image, factor):
        """Adjust image brightness"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def adjust_contrast(self, image, factor):
        """Adjust image contrast"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def augment_image(self, filename):
        """Apply augmentation to a single image"""
        # Read image
        image_path = os.path.join(self.input_dir, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error loading image: {filename}")
            return False
        
        try:
            name, ext = os.path.splitext(filename)
            augmented_images = []
            
            # Original image
            augmented_images.append((image, f"{name}_original{ext}"))
            
            # Rotations
            augmented_images.append((self.rotate_image(image, 90), f"{name}_rot90{ext}"))
            augmented_images.append((self.rotate_image(image, -90), f"{name}_rot-90{ext}"))
            
            # Horizontal flip
            augmented_images.append((self.horizontal_flip(image), f"{name}_flip{ext}"))
            
            # Brightness variations
            augmented_images.append((self.adjust_brightness(image, 0.8), f"{name}_dark{ext}"))
            augmented_images.append((self.adjust_brightness(image, 1.2), f"{name}_bright{ext}"))
            
            # Contrast variations
            augmented_images.append((self.adjust_contrast(image, 0.8), f"{name}_lowcontrast{ext}"))
            augmented_images.append((self.adjust_contrast(image, 1.2), f"{name}_highcontrast{ext}"))
            
            # Save augmented images
            for img, aug_filename in augmented_images:
                output_path = os.path.join(self.output_dir, aug_filename)
                cv2.imwrite(output_path, img)
            
            return True
            
        except Exception as e:
            print(f"Error augmenting {filename}: {str(e)}")
            return False
    
    def augment_all_images(self):
        """Augment all images in the input directory"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        success_count = 0
        total_count = 0
        
        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_count += 1
                if self.augment_image(filename):
                    success_count += 1
                    
        print(f"Successfully augmented {success_count} out of {total_count} images")
        print(f"Generated {success_count * 8} augmented images")

if __name__ == "__main__":
    augmentor = DataAugmentor()
    augmentor.augment_all_images()
