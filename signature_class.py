import os
import numpy as np
import json
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from paddleocr import PaddleOCR
from PIL import Image
import tensorflow as tf
from difflib import SequenceMatcher
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)

# Load the signature prediction model
def predict_class(image_path, model_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # List of class names
    class_names = ['signature found', 'signature not found']

    # Load and preprocess an image for testing
    def preprocess_image(image_path):
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    preprocessed_image = preprocess_image(image_path)

    prediction = model.predict(preprocessed_image)

    predicted_class = np.argmax(prediction)

    return class_names[predicted_class]

# Specify paths and parameters
img_path = '/Users/sivagar/Documents/projects/farrer_hos/projects/signature/FPH Test1_page-0007.jpg'
model_path = '/Users/sivagar/Documents/projects/farrer_hos/projects/signature/signature_model6.h5'
ocr_lang = 'en'
similarity_threshold = 0.9
prediction_threshold = 0.5


opencv_image = cv2.imread(img_path)
numpy_array = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # Convert color channels from BGR to RGB

# Load the OCR result
ocr = PaddleOCR(lang=ocr_lang)
result = ocr.ocr(numpy_array, rec=True)

# Load the JSON configuration
with open('/content/Config.json', 'r') as f:
    json_data = json.load(f)

# Get user input for document Type
user_input = "FORM-ICM-002-P01"

# Find the document with the user-input type
selected_document = None
for doc in json_data:
    if doc["Document_Type"] == user_input:
        selected_document = doc
        break

if selected_document:
    document_name = selected_document["Document_Type"]
    signatures = selected_document["Signature"]

    # Convert bounding box coordinates to the correct format
    boxes = [np.array(line[0]).reshape(-1, 2) for page in result for line in page]

    # Iterate through the found signatures in the selected document
    for signature in signatures:
        desired_text = signature["name"]
        signature_name = signature["signature_name"]
        vertical_align = signature["vertical_align"]
        horizontal_align = signature["horizontal_align"]

        found_boxes = []

        for box, line in zip(boxes, result[0]):
            text = line[1][0]
            similarity_ratio = SequenceMatcher(None, desired_text, text).ratio()

            if similarity_ratio >= similarity_threshold:
                found_boxes.append(box)

        if found_boxes:
            print("Document:", document_name)
            print("Signature:", signature_name)

            # Expand bounding boxes based on vertical_align and horizontal_align
            expanded_boxes = []
            for box in found_boxes:
                x_min, y_min = box.min(axis=0)
                x_max, y_max = box.max(axis=0)

                width = x_max - x_min
                height = y_max - y_min

                if "above" in vertical_align:
                    y_min -= height * vertical_align[1]

                if "below" in vertical_align:
                    y_max += height * vertical_align[1]

                if "left" in horizontal_align:
                    x_max += width * horizontal_align[2]

                if "right" in horizontal_align:
                    x_min -= width * horizontal_align[2]

                expanded_box = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
                expanded_boxes.append(expanded_box)

                #Save the cropped signature region
                signature_name_cleaned = signature_name.replace(" ", "_").replace("/", "_").replace("&", "_")
                filename = f'signature_{signature_name_cleaned}.png'

                # Convert NumPy array to a PIL image
                region = Image.fromarray(numpy_array[int(y_min):int(y_max), int(x_min):int(x_max)])

                # Save the cropped signature region
                region.save(filename)

                # Use the trained model to predict whether a signature is found
                predicted_class = predict_class(filename, model_path)

                if predicted_class == 'signature found':
                    print("Prediction: Signature Detected")
                else:
                    print("Prediction: Signature Not Detected")

                # ... (rest of the code)

else:
    print("Document not found.")
