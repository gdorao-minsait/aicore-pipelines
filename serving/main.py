import os
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO
import tempfile
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np

# Creates Flask serving engine
app = Flask(__name__)

model = None
appHasRunBefore = False
version = "v0.4"

def correct_image_orientation(image):
    try:
        # Check for EXIF data
        exif = image._getexif()
        if exif is not None:
            for tag, value in exif.items():
                if TAGS.get(tag) == 'Orientation':
                    orientation = value
                    break
        else:
            orientation = None

        # Apply the appropriate rotation
        if orientation is not None:
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print(f"Error reading EXIF data: {str(e)}")
    return image


def is_enclosed(box1, box2):
    """ Check if box1 is completely enclosed by box2. """
    # box format: [x1, y1, x2, y2]
    return (box1[0] >= box2[0] and box1[1] >= box2[1] and 
            box1[2] <= box2[2] and box1[3] <= box2[3])

def calculate_area(box):
    """ Calculate the area of a bounding box. 
    Box format: [x1, y1, x2, y2] """
    width = box[2] - box[0]  # x2 - x1
    height = box[3] - box[1]  # y2 - y1
    return max(0, width * height)  # Ensure non-negative area

def calculate_intersection(box1, box2):
    """ Calculate the intersection area between two boxes. """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Calculate the intersection dimensions
    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)

    return intersection_width * intersection_height

def is_almost_enclosed(box1, box2, threshold=0.9):
    """ Check if 90% of box1's area is inside box2.
    Box format: [x1, y1, x2, y2] """
    
    # Calculate areas of both boxes
    area_box1 = calculate_area(box1)
    area_box2 = calculate_area(box2)

    if area_box1 == 0 or area_box2 == 0:
        return False  # If either box has zero area, no enclosure is possible

    # Calculate the intersection area between box1 and box2
    intersection_area = calculate_intersection(box1, box2)

    # Check if 90% or more of box1's area is enclosed in box2
    if intersection_area >= threshold * area_box1:
        return True
    return False



@app.before_request
def init():
    """
    Load YOLO model for inference.
    """
    global model
    global appHasRunBefore

    if not appHasRunBefore:
        # Load the YOLO model from the /mnt/models directory
        model_path = '/mnt/models/yolo_model.pt'  # Path to your trained YOLO model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = YOLO(model_path)  # Load the YOLO model using ultralytics
        
        # Warm-up inference
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)  # Create a blank image
        model(dummy_image)  # Perform a dummy inference

        appHasRunBefore = True
        print("Model loaded successfully and warmed up.")

@app.route("/v1/check", methods=["GET"])
def status():
    global model
    if model is None:
        return f"Vessel Visionaries {version}. Flask Code: Model was not loaded.", 500
    else:
        return f"Vessel Visionaries {version}. Flask Code: Model loaded successfully.", 200


@app.route("/v1/detect", methods=["POST"])
def predict():
    global model
    if model is None:
        return "Flask Code: Model was not loaded.", 500

    if 'file' not in request.files:
        return "Flask Code: No file part in the request.", 400
    
    file = request.files['file']

    # Extract confidence threshold from request (as JSON or form data)
    try:
        confidence_threshold = float(request.form.get('confidence_threshold', 0.0))  # Default to 0.0 if not provided
    except ValueError:
        return "Flask Code: Invalid confidence threshold provided.", 400

    # Check if the file is an image
    try:
        image = Image.open(file)  # Open the uploaded image
        image = correct_image_orientation(image)  # Correct orientation
    except Exception as e:
        return f"Flask Code: Failed to process image: {str(e)}", 400

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp_file:
        temp_image_path = tmp_file.name
        image.save(temp_image_path)  # Save the uploaded image to the temp file
        
        # Perform YOLO inference on the saved temp image
        # results = model(temp_image_path, line_width=1, show_labels=True, save=True)  # TODO: save detect result in artifact for debugging purposes
        results = model(temp_image_path, save=False)  # Run inference on the image

    # Process results with confidence threshold and check for enclosed boxes
    class_counts = {}

    for result in results:
        boxes = []  # List to store bounding boxes above the confidence threshold

        # First, filter boxes by confidence threshold
        for box in result.boxes:
            confidence = box.conf  # Get the confidence score
            if confidence >= confidence_threshold:  # Apply the confidence threshold
                x1, y1, x2, y2 = box.xyxy[0]  # Extract the bounding box coordinates
                class_id = int(box.cls)  # Get the class ID
                class_name = model.names[class_id]  # Get the class name from the model
                
                # Append the box along with its coordinates, confidence, and class information
                boxes.append([x1, y1, x2, y2, confidence, class_name])

        # Now, filter out boxes that are almost completely enclosed within others (remove false positives)
        non_enclosed_boxes = []
        for i, box1 in enumerate(boxes):
            enclosed = False
            for j, box2 in enumerate(boxes):
                if i != j and is_almost_enclosed(box1[:4], box2[:4], 0.9):  # Check if box1 is enclosed by box2
                    enclosed = True
                    break
            if not enclosed:
                non_enclosed_boxes.append(box1)

        # Count the classes for the non-enclosed boxes
        for box in non_enclosed_boxes:
            class_name = box[5]  # Class name from the non-enclosed box
            class_counts[class_name] = class_counts.get(class_name, 0) + 1  # Increment class count


    # Return results as JSON
    return jsonify(class_counts), 200


if __name__ == "__main__":
    print("Serving Initializing")
    init()
    print(f"Serving Started. Vessel Visionaries {version}")
    app.run(host="0.0.0.0", debug=True, port=9001)
