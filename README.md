
# NeuroClust: Brain Tumor Segmentation using K-means

**NeuroClust** is a Streamlit-based web application that detects and segments brain tumors from MRI scans using K-means clustering. This project applies image processing techniques and machine learning to assist in the early detection of brain tumors.

## Features

- **Image Upload**: Upload brain scan images in `.jpg`, `.jpeg`, or `.png` format.
- **Preprocessing**: Image preprocessing using a bilateral filter to reduce noise while preserving edges.
- **K-means Clustering**: Segment the brain scan into different clusters using K-means clustering for tumor segmentation.
- **Tumor Detection**: Identify potential tumor regions based on contour area and solidity.
- **Adjustable Intensity**: Modify image intensity threshold using a slider to improve segmentation results.
- **Interactive UI**: Visualize original, preprocessed, and segmented images side by side for comparison.

## Technologies Used

- **Python**: Core programming language.
- **OpenCV**: For image processing and contour detection.
- **NumPy**: For efficient numerical operations.
- **Streamlit**: For creating the web application interface.
- **Pillow**: For handling image file formats in Python.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shubh-200/NeuroClust.git
   ```

2. Navigate to the project directory:

   ```bash
   cd NeuroClust
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run tumor_detect.py
   ```

## How it Works

1. **Upload an Image**: The user uploads a brain scan image.
2. **Preprocessing**: The image is preprocessed with a bilateral filter to remove noise while retaining edges.
3. **Clustering with K-means**: The app applies K-means clustering to segment the image into clusters.
4. **Tumor Detection**: The algorithm checks each cluster to identify potential tumors based on the contour area and shape solidity.
5. **Visualization**: The app displays preprocessed and processed images side by side for easy comparison.

## Usage

1. Start the application using Streamlit.
2. Upload an MRI image of a brain scan.
3. Adjust the intensity slider to fine-tune the segmentation.
4. View the original and tumor-detected images in the output panel.

## Future Improvements

- Improve the accuracy of tumor segmentation by incorporating more advanced algorithms such as U-Net or other deep learning methods.
- Add a feature to export segmented results as image files.
- Implement support for more file formats and larger datasets.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.


