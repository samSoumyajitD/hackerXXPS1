import cv2
import numpy as np
import os

def ensure_dirs(directories):
    """Ensure that the directories exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def table_detection(img_path, cropped_dir, table_detect_dir, bb_dir):
    """Detect tables in the image and save the results."""
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarize the image using Otsu's thresholding
    _, img_bin = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = cv2.bitwise_not(img_bin)
    
    # Calculate kernel lengths
    height, width = img_gray.shape
    kernel_length_v = width // 120
    kernel_length_h = width // 40
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
    vertical_lines_img = cv2.dilate(cv2.erode(img_bin, vertical_kernel, iterations=3), vertical_kernel, iterations=3)
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
    horizontal_lines_img = cv2.dilate(cv2.erode(img_bin, horizontal_kernel, iterations=3), horizontal_kernel, iterations=3)
    
    # Combine horizontal and vertical lines
    table_segment = cv2.addWeighted(vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
    table_segment = cv2.erode(cv2.bitwise_not(table_segment), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    
    # Find contours to detect table cells/segments
    contours, _ = cv2.findContours(table_segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # Filter contours to detect valid tables/boxes
        if (w > 80 and h > 20) and w > 3 * h:
            count += 1
            cropped = img[y:y + h, x:x + w]
            cropped_filename = os.path.join(cropped_dir, f"crop_{count}_{os.path.basename(img_path)}")
            cv2.imwrite(cropped_filename, cropped)
        
        # Draw rectangles around detected contours on the original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Save the processed images
    cv2.imwrite(os.path.join(table_detect_dir, f"table_detect_{os.path.basename(img_path)}"), table_segment)
    cv2.imwrite(os.path.join(bb_dir, f"bb_{os.path.basename(img_path)}"), img)

# Ensure output directories exist
form_dir = './forms/'
cropped_dir = './results/cropped/'
table_detect_dir = './results/table_detect/'
bb_dir = './results/bb/'
ensure_dirs([cropped_dir, table_detect_dir, bb_dir])

# Process all images in the form directory
image_files = [f for f in os.listdir(form_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
for img_file in image_files:
    img_path = os.path.join(form_dir, img_file)
    table_detection(img_path, cropped_dir, table_detect_dir, bb_dir)
