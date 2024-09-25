import numpy as np
from PIL import Image, ImageChops, ImageEnhance  # Added ImageEnhance here
import exifread
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel
import cv2

# Path to your image
image_path = 'test1.jpeg'

# --- 1. Metadata Extraction (EXIF) ---
def extract_exif(image_path):
    print("---- EXIF Data ----")
    try:
        with open(image_path, 'rb') as image_file:
            tags = exifread.process_file(image_file)
        for tag in tags.keys():
            if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
                print(f"{tag}: {tags[tag]}")
    except Exception as e:
        print(f"Error extracting EXIF data: {e}")

# --- 2. Time of Day Estimation using Pre-trained ResNet Model ---
def time_of_day_estimation(image_path):
    print("---- Time of Day Prediction ----")
    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path)
    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    print(output)  # Outputs raw tensor values (this can be refined with custom time labels)

# --- 4. Artistic Style Recognition using Pre-trained VGG16 Model ---
def artistic_style_recognition(image_path):
    print("---- Artistic Style Recognition ----")
    model = models.vgg16(weights='VGG16_Weights.DEFAULT')
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path)
    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    print(output)  # Outputs raw tensor values (this can be refined with labeled data for art styles)

# --- 5. Inferring the Artist’s Thoughts using CLIP Model ---
def infer_artist_thoughts(image_path):
    print("---- Artist’s Thoughts and Emotions Prediction ----")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path)

    prompts = [
        "What was the artist thinking?",
        "What emotions are reflected in this painting?",
        "What is the mood of this image?",
        "What does this painting represent?"
    ]

    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    for i, prompt in enumerate(prompts):
        print(f"{prompt}: {probs[0][i].item():.4f}")

# --- 6. Detecting Image Forgery using Error Level Analysis (ELA) ---
def detect_forgery_ela(image_path):
    print("---- Image Forgery Detection (ELA) ----")
    # Convert image to JPEG with 90% quality and compare differences
    img = Image.open(image_path).convert('RGB')
    img.save('temp_compressed.jpg', 'JPEG', quality=90)

    compressed_img = Image.open('temp_compressed.jpg')
    ela_img = ImageChops.difference(img, compressed_img)

    # Increase difference to make manipulations more visible
    extrema = ela_img.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1.0
    ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)

    # Show the ELA image
    ela_img.show()

    # Convert to numpy array for further analysis
    ela_img_np = np.asarray(ela_img)

    # Simple thresholding for potential tampering regions
    if np.mean(ela_img_np) > 10:  # This threshold can be adjusted
        print("Potential tampering detected based on ELA.")
    else:
        print("No significant tampering detected based on ELA.")

# Run all functions on the image
extract_exif(image_path)
time_of_day_estimation(image_path)
artistic_style_recognition(image_path)
infer_artist_thoughts(image_path)
detect_forgery_ela(image_path)  # Use ELA for forgery detection
