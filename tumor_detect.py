import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Function to preprocess the image using a bilateral filter
def preprocess_image(uploaded_image):
    image_array = np.array(uploaded_image)
    noisy_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    bilateral_denoised_img = cv2.bilateralFilter(noisy_image, d=9, sigmaColor=75, sigmaSpace=75)
    return bilateral_denoised_img

# Function to handle tumor detection logic
def tumor_part(c):
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area != 0:
        solidity = float(area) / hull_area
    else:
        solidity = 0
    return solidity > 0.5 and area > 2000

def blur_image(img):
    kernel = np.ones((5,5),np.float32)/25
    blur = cv2.filter2D(img,-1,kernel)
    return blur

def enhance(img):
    gray = cv2.equalizeHist(img)
    return gray

def threshold(img, b):
    _, thresh = cv2.threshold(img, b, 255, cv2.THRESH_BINARY)
    return thresh

def contours(img, org, b):
    img2 = threshold(img, b)
    cnts = cv2.findContours(img2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    img2 = RGB(img2)
    org = RGB(org)
    for c in cnts:
        if tumor_part(c):
            cv2.drawContours(org, [c], -1, (1, 255, 11), 2)
            cv2.drawContours(img2, [c], -1, (1, 255, 11), 2)
    return org, img2

def RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def k_means(img):
    Z = img.reshape((-1,1))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))

def show_images(gray, blur, seg, cont_org, cont_mask):
    res1 = np.hstack((blur, seg))
    res2 = np.hstack((cont_org, cont_mask))
    return np.vstack((res1, res2))

def process(img, b):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = blur_image(gray)
    seg = k_means(blur)
    cont_org, cont_mask = contours(seg, gray, b)
    seg = RGB(seg)
    blur = RGB(blur)
    gray = RGB(gray)
    return show_images(gray, blur, seg, cont_org, cont_mask)

# Streamlit app layout
st.title("Brain Tumor Detection using K-means")

# Image upload section
uploaded_image = st.file_uploader("Upload a brain scan image", type=["jpg", "jpeg", "png"])

# Intensity slider
b = st.slider("Adjust Intensity", min_value=50, max_value=240, value=130)

if uploaded_image:
    # Load the uploaded image
    image = Image.open(uploaded_image)

    # Preprocess the image using bilateral filter
    preprocessed_image = preprocess_image(image)

    # Convert back to BGR for further processing with OpenCV
    img_bgr = cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2BGR)

    # Process the image and generate the result
    result = process(img_bgr, b)

    # Display the input (preprocessed) and output (processed) images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(preprocessed_image, caption="Preprocessed Image (Bilateral Filter)", use_column_width=True)

    with col2:
        st.image(result, caption="Tumor Detection Result", use_column_width=True)
