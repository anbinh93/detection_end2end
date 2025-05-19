# Interactive Object Detection Web Application

This web application allows users to upload images and detect common objects using pre-trained Deep Learning models (YOLOv8 and DETR). Detected objects are highlighted with bounding boxes and labels. The application also features an optional image preprocessing module to enhance image quality.

## Features

*   **Multiple Image Upload:** Upload one or more images for detection.
*   **Model Selection:**
    *   Choose between various YOLOv8 model sizes (e.g., YOLOv8n, YOLOv8m).
    *   Select DETR (ResNet-101) model.
*   **Confidence Threshold:** Adjust the detection confidence threshold.
*   **Image Preprocessing (Optional):**
    *   **Denoise:** Apply Non-Local Means denoising.
    *   **Enhance Contrast:** Use CLAHE for adaptive contrast enhancement.
    *   **Sharpen:** Apply a light sharpening filter.
*   **Interactive Results:**
    *   View thumbnails of uploaded images.
    *   Display of processed image with bounding boxes and class labels.
    *   List of detected objects with confidence scores.
    *   Information on inference and preprocessing times.
*   **Save Results:** Option to save processed images with detections to the server.
*   **Responsive UI:** User interface adapts to different screen sizes.

## Project Structure
DETECTION_END2END/
├── CKPT/ # Directory for model checkpoint files (.pt, .pth)
│ └── detr-r101-2c7b67e5.pth
  └── YOLOv8n.pt
  └── YOLOv11n.pt
├── main.py # FastAPI application logic, API endpoints
├── model.py # Custom DETR model loader
├── preprocessing.py # Image preprocessing functions (OpenCV)
├── templates/
│ ├── index.html # Main application page
│ └── result.html # (Legacy, main UI now consolidated in index.html)
├── static/
│ └── style.css # CSS for styling
├── uploads/ # (Optional) Default directory for original uploads (if saved)
├── detected_images/ # Directory where processed images are saved
├── colors.json # (Optional) Custom color palette for bounding boxes
└── requirements.txt # Python dependencies

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd DETECTION_END2END
    ```

2.g  **Create a Virtual Environment (Recommended):**
    ```bash
    conda create -n "detection" python==3.10.12
    conda activate detection
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Model Checkpoints:**
    *   **YOLOv8:** Models like `yolov8n.pt`, `yolov8m.pt` will be automatically downloaded by the `ultralytics` library if not found in the `CKPT/` directory or system cache. You can also manually place `.pt` files in the `CKPT/` directory.
    *   **DETR:**
        *   The application attempts to load `detr-r101-2c7b67e5.pth` from the `CKPT/` directory by default. You can download this pre-trained DETR ResNet-101 model from a source like the original DETR repository or other model zoos.
        *   If the local `.pth` file is not found or fails to load, it will attempt to download `facebook/detr-resnet-101` from Hugging Face Hub.
    *   Create the `CKPT/` directory if it doesn't exist:
        ```bash
        mkdir CKPT
        ```


5.  **(Optional) Custom Colors:**
    Create a `colors.json` file in the root directory to define a custom color palette for bounding boxes. Example format:
    ```json
    [
      [255, 59, 59],
      [59, 130, 246],
      [34, 197, 94]
    ]
    ```
    If not found, default colors will be used.

## Running the Application

1.  Navigate to the project directory (`DETECTION_END2END/`).
2.  Run Uvicorn:
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `--reload`: Enables auto-reloading on code changes (for development).
    *   `--host 0.0.0.0`: Makes the app accessible from other devices on your network.
    *   `--port 8000`: Specifies the port.

3.  Open your web browser and go to `http://127.0.0.1:8000` (or your server's IP address if accessing remotely).

## Usage

1.  **Upload Images:** Click the "Upload Images" button and select one or more image files.
2.  **Select Model:** Choose your desired object detection model from the dropdown.
3.  **Adjust Confidence:** Set the confidence threshold using the slider. Detections below this threshold will be ignored.
4.  **Preprocessing (Optional):** Check the boxes for Denoise, Enhance Contrast, or Sharpen if you want to apply these enhancements before detection.
5.  **View Results:**
    *   Thumbnails of uploaded images appear in the "Uploaded Images" section.
    *   Click a thumbnail to display the processed image in the main view area.
    *   Bounding boxes and labels will be drawn on the image.
    *   Detection details (class, confidence), inference time, and preprocessing time (if applied) are shown below the image.
6.  **Process Images Button:** If you change model, confidence, or preprocessing options after images are uploaded, click "Process Images" to re-process all uploaded images with the new settings. Uploading new images or changing options also automatically triggers processing.
7.  **Save Processed:** Click to save all currently processed images (with detections drawn) to the `detected_images/` folder on the server.
8.  **Clear All:** Removes all uploaded images and results from the current session.

## Future Enhancements

*   Advanced control over preprocessing parameters.
*   Support for more model types or custom model uploads.
*   User authentication and persistent storage of results.
*   Task queuing for heavy processing to improve UI responsiveness.
*   Dockerization for easier deployment.