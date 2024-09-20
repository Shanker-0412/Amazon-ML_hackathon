import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import pytesseract
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

# Constants from utils.py
entity_unit_map = {
    "width": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
    "depth": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
    "height": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
    "item_weight": {"milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"},
    "maximum_weight_recommendation": {"milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"},
    "voltage": {"millivolt", "kilovolt", "volt"},
    "wattage": {"kilowatt", "watt"},
    "item_volume": {"cubic foot", "microlitre", "cup", "fluid ounce", "centilitre", "imperial gallon", "pint", "decilitre", "litre", "millilitre", "quart", "cubic inch", "gallon"}
}

# Load pre-trained DETR model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

def create_folders():
    """Create necessary folders if they don't exist."""
    folders = ['data', 'images', 'images/train', 'images/test', 'logs', 'output']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("Necessary folders have been created.")

def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def detect_text_areas(image):
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
    text_areas = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if model.config.id2label[label.item()] in ["text", "label", "sign"]:
            box = [round(i, 2) for i in box.tolist()]
            text_areas.append(box)
    
    return text_areas

def extract_text(image, text_areas):
    extracted_text = ""
    for box in text_areas:
        cropped_image = image.crop(box)
        text = pytesseract.image_to_string(cropped_image)
        extracted_text += text + " "
    return extracted_text.strip()

def extract_entity_value(text, entity_name):
    patterns = {
        "width": r'(\d+(?:\.\d+)?)\s*(cm|centimetre|foot|mm|millimetre|m|metre|inch|yard)',
        "depth": r'(\d+(?:\.\d+)?)\s*(cm|centimetre|foot|mm|millimetre|m|metre|inch|yard)',
        "height": r'(\d+(?:\.\d+)?)\s*(cm|centimetre|foot|mm|millimetre|m|metre|inch|yard)',
        "item_weight": r'(\d+(?:\.\d+)?)\s*(mg|milligram|kg|kilogram|µg|microgram|g|gram|oz|ounce|ton|pound)',
        "maximum_weight_recommendation": r'(\d+(?:\.\d+)?)\s*(mg|milligram|kg|kilogram|µg|microgram|g|gram|oz|ounce|ton|pound)',
        "voltage": r'(\d+(?:\.\d+)?)\s*(mV|millivolt|kV|kilovolt|V|volt)',
        "wattage": r'(\d+(?:\.\d+)?)\s*(kW|kilowatt|W|watt)',
        "item_volume": r'(\d+(?:\.\d+)?)\s*(ft³|cubic foot|µL|microlitre|cup|fl oz|fluid ounce|cL|centilitre|gal|imperial gallon|pt|pint|dL|decilitre|L|litre|mL|millilitre|qt|quart|in³|cubic inch|gallon)'
    }
    
    if entity_name in patterns:
        match = re.search(patterns[entity_name], text, re.IGNORECASE)
        if match:
            value, unit = match.groups()
            for std_unit in entity_unit_map[entity_name]:
                if std_unit.lower() in unit.lower():
                    return float(value), std_unit
    
    return None, None

def convert_to_appropriate_unit(entity, value, unit):
    
    if entity in ["width", "depth", "height"]:
        if unit in ["metre", "foot", "yard"]:
            return f"{value} {unit}"
        elif unit == "centimetre" and value >= 100:
            return f"{value / 100} metre"
        elif unit == "millimetre" and value >= 1000:
            return f"{value / 1000} metre"
        elif unit == "inch" and value >= 36:
            return f"{value / 36} yard"
    elif entity in ["item_weight", "maximum_weight_recommendation"]:
        if unit in ["ton", "pound"]:
            return f"{value} {unit}"
        elif unit == "kilogram" and value >= 1000:
            return f"{value / 1000} ton"
        elif unit == "gram" and value >= 1000:
            return f"{value / 1000} kilogram"
        elif unit in ["milligram", "microgram"] and value >= 1000000:
            return f"{value / 1000000} kilogram"
        elif unit == "ounce" and value >= 16:
            return f"{value / 16} pound"
    elif entity == "voltage":
        if unit == "volt" and value >= 1000:
            return f"{value / 1000} kilovolt"
        elif unit == "millivolt" and value >= 1000:
            return f"{value} volt"
    elif entity == "wattage":
        if unit == "watt" and value >= 1000:
            return f"{value / 1000} kilowatt"
    elif entity == "item_volume":
        if unit in ["litre", "gallon", "imperial gallon"]:
            return f"{value} {unit}"
        elif unit in ["millilitre", "centilitre", "decilitre"] and value >= 1000:
            return f"{value / 1000} litre"
        elif unit == "cubic foot" and value >= 7.48052:
            return f"{value / 7.48052} gallon"
        elif unit in ["cup", "pint", "quart"] and value >= 4:
            return f"{value / 4} gallon"
        elif unit == "fluid ounce" and value >= 128:
            return f"{value / 128} gallon"
    
    return f"{value} {unit}"
def process_image(image_url, entity_name):
    try:
        image = load_image_from_url(image_url)
        text_areas = detect_text_areas(image)
        text = extract_text(image, text_areas)
        value, unit = extract_entity_value(text, entity_name)
        if value is not None and unit is not None:
            return value, unit
    except Exception as e:
        print(f"Error processing image {image_url}: {str(e)}")
    return None, None

def prepare_data(df):
    X = []
    y = []
    for _, row in df.iterrows():
        value, unit = process_image(row['image_link'], row['entity_name'])
        if value is not None and unit is not None:
            X.append({
                'entity_name': row['entity_name'],
                'value': value,
                'unit': unit
            })
            y.append(row['entity_value'])
    return pd.DataFrame(X), y

def train_model(X, y):
    le_entity = LabelEncoder()
    le_unit = LabelEncoder()
    
    X['entity_encoded'] = le_entity.fit_transform(X['entity_name'])
    X['unit_encoded'] = le_unit.fit_transform(X['unit'])
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X[['entity_encoded', 'value', 'unit_encoded']], y)
    
    return model, le_entity, le_unit

def main():
    # Create necessary folders
    create_folders()

    # Load and prepare training data
    train_df = pd.read_csv('data/train.csv')
    X_train, y_train = prepare_data(train_df)
    
    # Train the model
    model, le_entity, le_unit = train_model(X_train, y_train)
    
    # Load test data
    test_df = pd.read_csv('data/test.csv')
    
    # Process test images and make predictions
    results = []
    for _, row in test_df.iterrows():
        value, unit = process_image(row['image_link'], row['entity_name'])
        if value is not None and unit is not None:
            entity_encoded = le_entity.transform([row['entity_name']])[0]
            unit_encoded = le_unit.transform([unit])[0]
            prediction = model.predict([[entity_encoded, value, unit_encoded]])[0]
            formatted_prediction = convert_to_appropriate_unit(row['entity_name'], prediction, unit)
        else:
            formatted_prediction = ""
        results.append({'index': row['index'], 'prediction': formatted_prediction})
    
    # Save results
    output_df = pd.DataFrame(results)
    output_df.to_csv('output/test_output.csv', index=False)
    
    print("Predictions saved to output/test_output.csv")

if __name__ == "__main__":
    main()