import requests
from PIL import Image
from io import BytesIO
import os
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

# Create directories for saving images if they don't exist
def create_directories():
    os.makedirs('images/train', exist_ok=True)
    os.makedirs('images/test', exist_ok=True)

# Load image from URL and save it to the specified directory (train/test)
def load_image_from_url(url, save_path):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))  # ResNet requires 224x224 images
        img.save(save_path)
        img_array = np.array(img)
        if img_array.shape != (224, 224, 3):
            img_array = img_array[:, :, :3]  # Ensure it's an RGB image
        return preprocess_input(np.expand_dims(img_array, axis=0))
    except:
        return None

# Load images for a batch of URLs and save them in the appropriate folder
def load_batch_images(urls, dataset_type):
    images = []
    for i, url in enumerate(urls):
        save_path = f'images/{dataset_type}/image_{i}.jpg'
        img = load_image_from_url(url, save_path)
        if img is not None:
            images.append(img)
    return np.vstack(images)  # Return as a single batch
