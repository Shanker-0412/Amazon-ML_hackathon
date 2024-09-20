import random
import csv
from collections import defaultdict

# Entity-Unit map (as provided in the appendix)
entity_unit_map = {
    "width": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
    "depth": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
    "height": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
    "item_weight": {"milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"},
    "maximum_weight_recommendation": {"milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"},
    "voltage": {"millivolt", "kilovolt", "volt"},
    "wattage": {"kilowatt", "watt"},
    "item_volume": {"cubic foot", "microlitre", "cup", "fluid ounce", "centilitre", "imperial gallon", 
                    "pint", "decilitre", "litre", "millilitre", "quart", "cubic inch", "gallon"}
}

# Define realistic value ranges for each entity
value_ranges = {
    "width": (1, 500),
    "depth": (1, 500),
    "height": (1, 500),
    "item_weight": (0.1, 1000),
    "maximum_weight_recommendation": (1, 5000),
    "voltage": (1, 240),
    "wattage": (1, 2000),
    "item_volume": (0.1, 100)
}

# Define probability of not generating a prediction
NO_PREDICTION_PROB = 0.1

def generate_prediction(entity):
    if random.random() < NO_PREDICTION_PROB:
        return ""
    
    unit = random.choice(list(entity_unit_map[entity]))
    min_val, max_val = value_ranges[entity]
    
    if "metre" in unit or "litre" in unit:
        value = round(random.uniform(min_val/100, max_val/100), 2)
    elif unit in ["kilogram", "kilovolt", "kilowatt"]:
        value = round(random.uniform(min_val/1000, max_val/1000), 2)
    else:
        value = round(random.uniform(min_val, max_val), 2)
    
    return f"{value} {unit}"

def generate_predictions(test_file, output_file):
    entity_counts = defaultdict(int)
    
    # Read the test data and count entities
    with open(test_file, mode='r') as file:
        reader = csv.DictReader(file)
        test_data = list(reader)
        for row in test_data:
            entity_counts[row['entity_name']] += 1
    
    # Calculate prediction probabilities based on entity frequency
    total_entities = sum(entity_counts.values())
    pred_probs = {entity: count / total_entities for entity, count in entity_counts.items()}
    
    # Create output file and write the header
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["index", "prediction"])
        
        # Loop through each row in the test data
        for row in test_data:
            index = row['index']
            entity_name = row['entity_name']
            
            # Generate prediction for this row
            if random.random() < pred_probs[entity_name]:
                prediction = generate_prediction(entity_name)
            else:
                prediction = ""
            
            # Write the index and prediction to the output file
            writer.writerow([index, prediction])

    print(f"Predictions successfully saved to {output_file}")

# Specify the test input file and output file
test_file = "dataset/test.csv"  # Replace with actual test file path
output_file = "test_out.csv"

# Generate predictions
generate_predictions(test_file, output_file)