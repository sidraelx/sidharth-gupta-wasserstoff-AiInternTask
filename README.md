# sidharth-gupta-wasserstoff-AiInternTask
Ai Internship Task for Wasserstoff Innovation and Learning Labs

# Image Segmentation and Object Recognition

This project demonstrates how to perform image segmentation and object recognition using pre-trained models in PyTorch.

## Installation

Install the required libraries using pip
```
bash pip install torch torchvision transformers opencv-python-headless matplotlib pandas easyocr
```

## Usage

1. **Load the Pascal VOC dataset:**
   - The code downloads and prepares the Pascal VOC dataset for object detection.

2. **Visualize images and bounding boxes:**
   - The `visualize_voc_image` function displays images with bounding boxes around detected objects.

3. **Perform image segmentation:**
   - A pre-trained Mask R-CNN model is used to segment images, identifying different objects.

4. **Extract and save objects:**
   - The `extract_objects` function extracts individual objects from segmented images and saves them as separate files.

5. **Identify and describe objects:**
   - The CLIP model is used to identify and describe the extracted objects.

6. **Extract text from objects:**
   - The EasyOCR library is used to extract text from the object images.

7. **Summarize object attributes:**
   - The `summarize_attributes` function compiles a summary of each object's description, extracted text, and file path.

8. **Map data and generate output:**
   - The `map_data` function organizes the object data by master ID.
   - The `generate_output` function displays the original image, a summary table of object attributes, and annotations on the image.

## Example

The provided code includes an example of how to use these functions to process an image from the Pascal VOC dataset.

## Notes

- Ensure you have a GPU available for optimal performance.
- Adjust the code and model paths as needed for your specific environment.
