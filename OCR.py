import cv2
import pytesseract
import numpy as np
import streamlit as st
from PIL import Image


# def preprocess_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     enhanced = cv2.equalizeHist(blurred)
#     edged = cv2.Canny(enhanced, 50, 150)
#     return edged


def preprocess_image(image):
    if len(image.shape) == 2:  # Grayscale image
        gray = image
    elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:  # BGRA image
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Unsupported image format")

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = cv2.equalizeHist(blurred)
    edged = cv2.Canny(enhanced, 50, 150)
    return edged


def find_plate_contour(edged, image):
    contours, _ = cv2.findContours(
        edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(
            contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 5:
                return image[y:y+h, x:x+w]
    return None


def extract_number_plate_text(plate_image):
    if plate_image is None:
        return "Number plate not detected"
    plate_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    plate_gray = cv2.adaptiveThreshold(
        plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    plate_gray = cv2.morphologyEx(plate_gray, cv2.MORPH_CLOSE, kernel)
    text = pytesseract.image_to_string(
        plate_gray, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    return text.strip()


def main():
    st.title("Car Number Plate Detection")
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "png", "jpeg", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        edged = preprocess_image(image)
        plate_image = find_plate_contour(edged, image)
        number_plate_text = extract_number_plate_text(plate_image)

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("### Detected Number Plate:")
        st.write(f"**{number_plate_text}**")

        if plate_image is not None:
            st.image(plate_image, caption="Detected Plate",
                     use_column_width=True)


if __name__ == "__main__":
    main()
