import os
import glob
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path, min_area_threshold=1000, padding=20):
    """
    Processes a single image by adding a black border (padding),
    then computing a convex hull and minimum enclosing circle for the white circle.
    Returns a matplotlib figure and the computed hull circularity.
    """
    # Read image and add a black border to avoid border artifacts
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not load image: {image_path}")
    padded = cv2.copyMakeBorder(image, padding, padding, padding, padding, 
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # Convert to grayscale and blur to reduce noise
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Otsu thresholding
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Perform morphological closing to fill in gaps and smooth the shape
    kernel_close = np.ones((15, 15), np.uint8)
    thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Find external contours
    contours, _ = cv2.findContours(thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No contours detected in {image_path} after closing.")
    
    # Filter out small contours by area
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area_threshold]
    if not valid_contours:
        raise ValueError(f"No valid contours found in {image_path} after area filtering.")
    
    # Assume the desired white circle is the largest contour by area
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Compute convex hull to remove indentations
    hull = cv2.convexHull(largest_contour)
    
    # Compute area and perimeter of the hull for circularity
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    if hull_perimeter == 0:
        raise ValueError(f"Hull perimeter is zero in {image_path}, cannot compute circularity.")
    hull_circularity = (4 * np.pi * hull_area) / (hull_perimeter ** 2)
    
    # Get minimum enclosing circle for the hull
    (x, y), radius = cv2.minEnclosingCircle(hull)
    center = (int(x), int(y))
    radius = int(radius)
    
    # Draw results on a copy of the padded image
    output = padded.copy()
    cv2.drawContours(output, [hull], -1, (0, 255, 0), 2)
    # Draw the enclosing circle in vibrant red
    cv2.circle(output, center, radius, (0, 0, 255), 2)
    cv2.putText(output, f"Circ: {hull_circularity:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Create a figure with two subplots: left: post-closing thresholded, right: final result
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(thresh_closed, cmap='gray')
    axes[0].set_title("Post-Closing Thresholded")
    axes[0].axis("off")
    
    axes[1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Convex Hull & Circle\nCircularity: {hull_circularity:.3f}")
    axes[1].axis("off")
    
    fig.tight_layout()
    
    return fig, hull_circularity

def main(input_folder="."):
    """
    Processes all images with suffix "_cropped.tif" in the input_folder,
    saves each processed output as a PDF, and writes a CSV with image name and circularity.
    """
    image_paths = glob.glob(os.path.join(input_folder, "*_crop.tif"))
    if not image_paths:
        print(f"No images with suffix '_crop.tif' found in '{input_folder}'.")
        return
    
    results = []
    
    # Create a folder to store PDFs
    pdf_folder = os.path.join(input_folder, "pdf_outputs")
    os.makedirs(pdf_folder, exist_ok=True)
    
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            fig, circ_value = process_image(img_path, min_area_threshold=1000, padding=20)
            results.append((base_name, circ_value))
            
            # Save the figure as a PDF
            pdf_filename = f"{base_name}.pdf"
            pdf_path = os.path.join(pdf_folder, pdf_filename)
            fig.savefig(pdf_path)
            plt.close(fig)
            
            print(f"Processed {img_path} | Circularity: {circ_value:.3f}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Write results to CSV
    csv_path = os.path.join(input_folder, "results.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["ImageName", "HullCircularity"])
        for row in results:
            writer.writerow([row[0], f"{row[1]:.3f}"])
    
    print(f"\nResults saved to: {csv_path}")
    print(f"PDF outputs saved in: {pdf_folder}")

if __name__ == "__main__":
    main(input_folder=".")
