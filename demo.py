import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Function to denoise an image and return processed images
def denoise_image(uploaded_image):
    image_array = np.array(uploaded_image)

    # Convert the image to RGB format if not already
    noisy_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # Apply Gaussian, Median, and Bilateral filters
    gaussian_denoised_img = cv2.GaussianBlur(noisy_image, ksize=(5, 5), sigmaX=0)
    median_denoised_img = cv2.medianBlur(noisy_image, ksize=5)
    bilateral_denoised_img = cv2.bilateralFilter(noisy_image, d=9, sigmaColor=75, sigmaSpace=75)

    # Return all processed images in a dictionary
    return {
        "Noisy Image": noisy_image,
        "Gaussian Blur": gaussian_denoised_img,
        "Median Blur": median_denoised_img,
        "Bilateral Filter": bilateral_denoised_img
    }

# Streamlit app layout
st.title("Brain Tumor Detection")

# Image upload section (allow multiple uploads)
uploaded_images = st.file_uploader("Upload brain scan images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_images:
    for uploaded_image in uploaded_images:
        # Display each uploaded image in a new row
        st.write(f"### Processing image: {uploaded_image.name}")
        
        # Load the uploaded image as a PIL image
        image = Image.open(uploaded_image)

        # Preprocess the image using denoise_image function
        processed_images = denoise_image(image)

        # Create a new row (columns) for each image
        col1, col2, col3, col4 = st.columns(4)

        # Display the original and processed images in the columns
        with col1:
            st.image(processed_images["Noisy Image"], caption="Noisy Image", use_column_width=True)
        with col2:
            st.image(processed_images["Gaussian Blur"], caption="Gaussian Blur", use_column_width=True)
        with col3:
            st.image(processed_images["Median Blur"], caption="Median Blur", use_column_width=True)
        with col4:
            st.image(processed_images["Bilateral Filter"], caption="Bilateral Filter", use_column_width=True)
