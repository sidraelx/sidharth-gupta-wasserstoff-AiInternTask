import streamlit as st
from PIL import Image
import io
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from transformers import CLIPProcessor, CLIPModel
import easyocr
import numpy as np

# Load models and processors
device = "cuda" if torch.cuda.is_available() else "cpu"
maskrcnn_model = maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
ocr_reader = easyocr.Reader(['en'])

def preprocess_image(image):
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)

def segment_image(image):
    input_image = preprocess_image(image)
    with torch.no_grad():
        output = maskrcnn_model(input_image)
    return output

def extract_objects(image, output):
    image = np.array(image)
    object_metadata = []
    masks = output[0]['masks'].mul(255).byte().cpu().numpy()
    for idx, mask in enumerate(masks):
        mask = mask[0]
        object_image = image * np.stack([mask, mask, mask], axis=2)
        object_path = f'object_{idx}.png'
        Image.fromarray(object_image).save(object_path)
        object_metadata.append({
            'object_id': str(idx),
            'object_path': object_path
        })
    return object_metadata

def identify_objects(metadata):
    descriptions = []
    for item in metadata:
        object_image = Image.open(item['object_path'])
        inputs = clip_processor(text=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'], images=object_image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        description = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'][probs.argmax()]
        item['description'] = description
        descriptions.append(item)
    return descriptions

def extract_text(metadata):
    for item in metadata:
        object_image = Image.open(item['object_path'])
        text = ocr_reader.readtext(np.array(object_image), detail=0, paragraph=True)
        item['extracted_text'] = " ".join(text)
    return metadata

def summarize_attributes(metadata):
    for item in metadata:
        attributes = {
            'description': item['description'],
            'extracted_text': item.get('extracted_text', ''),
            'object_path': item['object_path']
        }
        item['attributes'] = attributes
    return metadata

def map_data(metadata):
    mapped_data = {}
    for item in metadata:
        if item['object_id'] not in mapped_data:
            mapped_data[item['object_id']] = []
        mapped_data[item['object_id']].append(item['attributes'])
    return mapped_data

st.title('Image Processing App')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Processing...")
    output = segment_image(image)
    metadata = extract_objects(image, output)
    identified_metadata = identify_objects(metadata)
    text_metadata = extract_text(identified_metadata)
    summarized_metadata = summarize_attributes(text_metadata)
    mapped_data = map_data(summarized_metadata)
    st.write(mapped_data)
