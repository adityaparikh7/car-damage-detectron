import cv2
import pytesseract
import numpy as np
import streamlit as st
from PIL import Image
import io


# # def preprocess_image(image):
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# #     enhanced = cv2.equalizeHist(blurred)
# #     edged = cv2.Canny(enhanced, 50, 150)
# #     return edged


# def preprocess_image(image):
#     if len(image.shape) == 2:  # Grayscale image
#         gray = image
#     elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR image
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     elif len(image.shape) == 3 and image.shape[2] == 4:  # BGRA image
#         gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
#     else:
#         raise ValueError("Unsupported image format")

#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     enhanced = cv2.equalizeHist(blurred)
#     edged = cv2.Canny(enhanced, 50, 150)
#     return edged


# def find_plate_contour(edged, image):
#     contours, _ = cv2.findContours(
#         edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     for contour in contours:
#         approx = cv2.approxPolyDP(
#             contour, 0.02 * cv2.arcLength(contour, True), True)
#         if len(approx) == 4:
#             x, y, w, h = cv2.boundingRect(approx)
#             aspect_ratio = w / float(h)
#             if 2 < aspect_ratio < 5:
#                 return image[y:y+h, x:x+w]
#     return None


# def extract_number_plate_text(plate_image):
#     if plate_image is None:
#         return "Number plate not detected"
#     plate_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
#     plate_gray = cv2.adaptiveThreshold(
#         plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     kernel = np.ones((3, 3), np.uint8)
#     plate_gray = cv2.morphologyEx(plate_gray, cv2.MORPH_CLOSE, kernel)
#     text = pytesseract.image_to_string(
#         plate_gray, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
#     return text.strip()


# def main():
#     st.title("Car Number Plate Detection")
#     uploaded_file = st.file_uploader(
#         "Upload an image", type=["jpg", "png", "jpeg", "webp"])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         image = np.array(image)
#         edged = preprocess_image(image)
#         plate_image = find_plate_contour(edged, image)
#         number_plate_text = extract_number_plate_text(plate_image)

#         st.image(image, caption="Uploaded Image", use_column_width=True)
#         st.write("### Detected Number Plate:")
#         st.write(f"**{number_plate_text}**")

#         if plate_image is not None:
#             st.image(plate_image, caption="Detected Plate",
#                      use_column_width=True)


# if __name__ == "__main__":
#     main()



def preprocess_image(image):
    if len(image.shape) == 2:  # Grayscale image
        gray = image
    elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:  # BGRA image
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Unsupported image format")

    # Apply adaptive thresholding instead of simple blurring
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try CLAHE (Contrast Limited Adaptive Histogram Equalization) instead of simple equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Bilateral filter preserves edges while removing noise
    bilateral = cv2.bilateralFilter(enhanced, 11, 17, 17)
    
    # Edge detection
    edged = cv2.Canny(bilateral, 30, 200)
    
    # Use morphological operations to close gaps in edges
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    
    return edged


def find_plate_contour(edged, image):
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_contour = None
    plate_image = None
    max_score = 0
    
    for contour in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # Check if it has 4 corners (like a rectangle)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            
            # Calculate aspect ratio of the contour
            aspect_ratio = float(w) / h
            
            # Most license plates have aspect ratio between 2 and 5
            if 1.5 <= aspect_ratio <= 5.5:
                # Calculate contour area and use it as part of scoring
                area = cv2.contourArea(contour)
                area_score = area / (image.shape[0] * image.shape[1])  # Normalized area
                
                # Calculate how rectangular the shape is
                rect_area = w * h
                extent = float(area) / rect_area
                
                # Calculate score based on aspect ratio and rectangularity
                score = extent * (1 - abs((aspect_ratio - 3.5) / 2))  # 3.5 is ideal aspect ratio
                
                if score > max_score:
                    max_score = score
                    plate_contour = approx
                    plate_image = image[y:y+h, x:x+w]
    
    return plate_image

def extract_number_plate_text(plate_image):
    if plate_image is None:
        return "Number plate not detected"
    
    # Convert to grayscale if not already
    if len(plate_image.shape) == 3:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_image
    
    # Resize for better OCR if plate is too small
    height, width = gray.shape
    if height < 50:
        scale_factor = 50 / height
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # Apply multiple preprocessing techniques and combine results
    results = []
    
    # Method 1: Adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    text1 = pytesseract.image_to_string(
        thresh, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    results.append(text1.strip())
    
    # Method 2: Otsu's thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text2 = pytesseract.image_to_string(
        otsu, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    results.append(text2.strip())
    
    # Method 3: Original image with different PSM
    text3 = pytesseract.image_to_string(
        gray, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    results.append(text3.strip())
    
    # Choose the result with the most alphanumeric characters
    final_text = max(results, key=lambda x: sum(c.isalnum() for c in x))
    
    return final_text.strip()

def main():
    st.set_page_config(page_title="Car Number Plate Detection", layout="wide")
    
    st.title("Car Number Plate Detection")
    st.markdown("Upload a car image to detect and extract the license plate text.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "webp"])
        
        if uploaded_file is not None:
            with st.spinner("Processing image..."):
                # Display original image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Convert to numpy array for processing
                image_np = np.array(image)
                
                # Save processing steps to display them
                edged = preprocess_image(image_np)
                plate_image = find_plate_contour(edged, image_np)
                number_plate_text = extract_number_plate_text(plate_image)
    
    with col2:
        if uploaded_file is not None:
            # Display results
            st.subheader("Processing Results")
            
            # Show edges for debugging
            st.image(edged, caption="Edge Detection", use_column_width=True)
            
            if plate_image is not None:
                st.image(plate_image, caption="Detected License Plate", use_column_width=True)
                
                # Display the extracted text with confidence
                st.markdown("### Detected License Plate Text")
                st.markdown(f"<h2 style='text-align: center;'>{number_plate_text}</h2>", unsafe_allow_html=True)
            else:
                st.error("No license plate detected in the image.")
                
            # Add download button for processed image
            if plate_image is not None:
                pil_plate = Image.fromarray(plate_image)
                buf = io.BytesIO()
                pil_plate.save(buf, format="PNG")
                st.download_button(
                    label="Download License Plate",
                    data=buf.getvalue(),
                    file_name="license_plate.png",
                    mime="image/png"
                )