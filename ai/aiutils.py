import os
import warnings
warnings.filterwarnings('ignore')
from keras_vggface.utils import preprocess_input
import numpy as np
import pandas as pd
import cv2
from django.core.exceptions import ValidationError
from keras_vggface.utils import preprocess_input
from PIL import Image
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
import time
from PIL import Image
from PIL.ExifTags import TAGS
from pillow_heif import register_heif_opener
from load_engines import resnet50_model, senet50_model, vgg16_model, w_detect_face, class_labels
register_heif_opener()
from typing import Tuple, Optional
import ast

def save_face_from_src(image_path: str, box: str, save_path: str, size: Tuple[int, int] = (300, 500), margins: Tuple[int, int] = (50, 50)) -> Optional[Image.Image]:
    # Check if the image file exists
    if not os.path.isfile(image_path):
        logging.error(f"Image file does not exist: {image_path}")
        return None
    # Parse the box string into a list of integers
    try:
        x, y, w, h = ast.literal_eval(box)
    except ValueError as e:
        logging.error(f"Failed to parse box string: {e}")
        return None
    logging.info(f"Parsed box: {box}")
    # Calculate box coordinates with margin
    x1 = max(0, x - margins[0])  # Ensure x1 is not less than 0
    y1 = max(0, y - margins[1])  # Ensure y1 is not less than 0
    x2 = x + w + margins[0]  
    y2 = y + h + margins[1]  
    try:
        with Image.open(image_path).convert("RGB") as img:
            # Make sure x2 and y2 do not exceed image dimensions
            x2 = min(img.width, x2)  # Ensure x2 is not more than image width
            y2 = min(img.height, y2)  # Ensure y2 is not more than image height
            logging.info(f"Calculated box coordinates: {(x1, y1, x2, y2)}")
            # Crop image
            img_cropped = img.crop((x1, y1, x2, y2))
            logging.info(f"Cropped image size: {img_cropped.size}")
            # Resize image
            img_resized = img_cropped.resize(size)
            logging.info(f"Resized image size: {img_resized.size}")
            # Save resized image
            img_resized.save(save_path)
            logging.info(f"Successfully processed and saved image: {save_path}")
            return img_resized
    except Exception as e:
        logging.error(f"Failed to process image: {e}")
        return None

def correct_image_orientation(image_path):
    try:   
        with Image.open(image_path) as image:
            exif = None
            if hasattr(image, "_getexif") and image._getexif() is not None:
                exif = {TAGS[key]: value for key, value in image._getexif().items() if key in TAGS and isinstance(TAGS[key], str)}
                # Get the orientation tag (key: "Orientation")
                orientation = exif.get("Orientation")
                # Rotate the image based on the orientation tag
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
                image.save(image_path)
            image_np = np.array(image)
            return image_np
    except (AttributeError, KeyError, IndexError, OSError, IOError) as e:
        # Cases: image don't have getexif
        raise ValidationError(f"Image could not be oriented correctly. Error: {str(e)}")

def detect_faces(img_cv2, min_size, confidence_threshold):
    start_time = time.time()
    detected_faces = []
    faces = w_detect_face(img_cv2)
    logger.debug(f"Raw Detected {len(faces)} faces.")
    if faces:
        for face in faces:
            box, confidence = face['box'], face['confidence']
            if confidence < confidence_threshold:
                logger.debug(f"Face confidence {confidence} lower than threshold {confidence_threshold}. Skipping.")
                continue
            x, y, width, height = box
            if width < min_size[0] or height < min_size[1]:
                logger.debug(f"Face size {width, height} smaller than minimum size {min_size}. Skipping.")
                continue
            face_pixels = img_cv2[y: y + height, x: x + width]
            # face_check = w_detect_face(face_pixels)
            if True: # face_check disabled
                detected_faces.append({
                    'face_pixels': face_pixels,            
                    'box': box,
                    'confidence': confidence,
                })
    logger.debug(f"Detection took {(time.time() - start_time):.2f} seconds. Detected {len(detected_faces)} faces.")
    return detected_faces


def extract_faces(medsession_dir, min_size=(50, 50), confidence_threshold=0.6):
    start_time = time.time()
    faces_df = []
    face_count = 0
    for file_name in os.listdir(medsession_dir):
        file_path = os.path.join(medsession_dir, file_name)
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File path does not exist: {file_path}")
            if os.path.getsize(file_path) == 0:
                raise ValueError(f"File is empty: {file_path}")
            img = correct_image_orientation(file_path)
            img = np.array(img, dtype='uint8')
            img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            logger.debug(f'image: {file_path}, size: {len(np.array(img))}')
            detected_faces = detect_faces(img_cv2, min_size, confidence_threshold)
            if not detected_faces:
                logger.debug(f'No faces detected in the image: {file_path}.')
                continue
            for face in detected_faces:
                face['image_path'] = file_path
                faces_df.append(face)
            face_count += len(detected_faces)
            logger.debug(f'{len(faces_df)} faces extracted from image: {file_path}.')
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            continue
    faces_df = pd.DataFrame(faces_df)
    logger.debug(f"Extracted {face_count} faces from {len(os.listdir(medsession_dir))} files in {(time.time() - start_time):.2f} seconds.")
    return faces_df

def predict_in_batches(face_pixels, model, batch_size=64):
    start_time = time.time()
    predictions = []
    for i in range(0, face_pixels.shape[0], batch_size):
        batch = face_pixels[i:i+batch_size]
        batch_predictions = model.predict(batch)
        predictions.extend(batch_predictions)
    end_time = time.time()
    logger.debug(f"Prediction took {(end_time - start_time):.2f} seconds. Made {len(predictions)} predictions.")
    return np.array(predictions)

def query_models(faces_df, confidence_threshold=0.98):
    # Make a copy of faces_df and reset its index
    person_df = faces_df.copy()
    person_df.reset_index(drop=True, inplace=True)

    # Initiate columns for each model
    for model_name in ['resnet50', 'senet50', 'vgg16']:
        person_df[model_name + '_label'] = ''
        person_df[model_name + '_confidence'] = 0.0

    # Convert face pixel data to numpy array for model prediction
    std_size = (224, 224)
    face_pixels = [Image.fromarray(face).resize(std_size) for face in person_df['face_pixels']]
    face_pixels = np.stack(face_pixels)
    face_pixels = face_pixels.astype('float64')
    face_pixels = preprocess_input(face_pixels)   

    models = {
        'resnet50': resnet50_model,
        'senet50': senet50_model,
        'vgg16': vgg16_model,
    }

    # Load each model and make predictions
    for model_name, model in models.items():
        logger.debug(f"Making predictions with {model_name} model...")
        predictions = predict_in_batches(face_pixels, model, batch_size=64)

        logger.debug("Updating DataFrame with predictions...")
        class_confidences = np.max(predictions, axis=1)
        class_indices = [np.argmax(pred) if conf >= confidence_threshold else None for pred, conf in zip(predictions, class_confidences)]

        person_df[model_name + '_label'] = [class_labels[i] if i is not None else None for i in class_indices]
        person_df[model_name + '_confidence'] = class_confidences

        above_threshold_predictions = sum(conf >= confidence_threshold for conf in class_confidences)
        logger.debug(f"Made {above_threshold_predictions} predictions above the confidence threshold of {confidence_threshold}.")
        logger.debug(f"Average confidence of predictions: {np.mean([conf for conf in class_confidences if conf >= confidence_threshold]):.4f}.")

        logger.debug(f"Completed predictions with {model_name} model.")
    return person_df


import numpy as np
from collections import Counter
import time
import logging

# Set up logger
logger = logging.getLogger(__name__)

def elect_answer(person_df, weights=[0.1, 0.3, 0.5]):
    start_time = time.time()
    # New column for elected label
    person_df['elected_label'] = ''    
    for i, person in person_df.iterrows():
        labels = [person['vgg16_label'], person['resnet50_label'], person['senet50_label']]
        confidences = np.array([person['vgg16_confidence'], person['resnet50_confidence'], person['senet50_confidence']])        

        # Majority voting
        label_counts = Counter(labels)
        max_count_label = label_counts.most_common(1)[0][0]
        print(f"Majority voting label for row {i}: {max_count_label}")
        logger.debug(f"Majority voting label for row {i}: {max_count_label}")

        # Weighted voting
        weighted_confidences = confidences * weights
        max_weighted_label = labels[np.argmax(weighted_confidences)]
        print(f"Weighted voting label for row {i}: {max_weighted_label}")
        logger.debug(f"Weighted voting label for row {i}: {max_weighted_label}")

        # Average probabilities
        avg_probability_label = labels[np.argmax(confidences.mean(axis=0))]
        print(f"Average probabilities label for row {i}: {avg_probability_label}")
        logger.debug(f"Average probabilities label for row {i}: {avg_probability_label}")

        # Final decision
        final_decision = Counter([max_count_label
                                  , max_weighted_label
                                  , avg_probability_label]).most_common(1)[0][0]
        # person_df.loc[i, 'elected_label'] = final_decision
        person_df.loc[i, 'elected_label'] = final_decision

        print(f"Elected label for row {i}: {final_decision}")
        logger.debug(f"Elected label for row {i}: {final_decision}")

    logger.info(f"Elected answers for {len(person_df)} rows in {(time.time() - start_time):.2f} seconds.")
    print(f"Elected answers for {len(person_df)} rows in {(time.time() - start_time):.2f} seconds.")
    return person_df


def elect_answer2(person_df):
    start_time = time.time()
    # New column for elected label
    person_df['elected_label'] = ''    
    for i, person in person_df.iterrows():
        # logger.debug(f"Processing row {i}...")
        
        labels = [person['resnet50_label'], person['senet50_label'], person['vgg16_label']]
        confidences = [person['resnet50_confidence'], person['senet50_confidence'], person['vgg16_confidence']]        
        # Count the occurrences of each label
        label_counts = {label: labels.count(label) for label in labels}
        # logger.debug(f"label counts: {label_counts}")        
        # Find the label(s) with the maximum count
        max_count = max(label_counts.values())
        max_labels = [label for label, count in label_counts.items() if count == max_count]
        # logger.debug(f"max labels: {max_labels}")        
        if len(max_labels) == 1:
            # If there's one label with the maximum count, use it
            person_df.loc[i, 'elected_label'] = max_labels[0]
        else:
            # If there's a tie, use the label with the highest average confidence
            avg_confidences = [np.mean([confidences[j] for j in range(3) if labels[j] == label]) for label in max_labels]
            person_df.loc[i, 'elected_label'] = max_labels[np.argmax(avg_confidences)]
        # logger.debug(f"Elected label for row {i}: {person_df.loc[i, 'elected_label']}")
    logger.info(f"Elected answers for {len(person_df)} rows in {(time.time() - start_time):.2f} seconds.")
    return person_df

def remove_duplicates(elected):
    start_time = time.time()
    logger.debug(f"Total records before removing duplicates: {len(elected)}")
    
    # Detect duplicates
    duplicates = elected[elected.duplicated(subset='elected_label', keep=False)]
    logger.debug("Duplicates:")
    logger.debug(duplicates)
    
    # Add column for area of bounding box
    elected['box_area'] = elected['box'].apply(lambda b: b[2]*b[3])
    
    # Sort by box_area in descending order so that duplicates with the largest area come first
    elected.sort_values('box_area', ascending=False, inplace=True)
    
    # Remove duplicates, keeping the first one (which, due to the sorting, will be the one with the largest area)
    unique_persons = elected.drop_duplicates(subset='elected_label')
    logger.debug(f"Total records after removing duplicates: {len(unique_persons)}")
    
    # Split 'elected_label' and create a new column 'name' 
    unique_persons['fullname'] = unique_persons['elected_label'].apply(lambda x: x.split('_')[1] if x and '_' in x else x)
    # Sort DataFrame by 'name' in ascending order
    unique_persons_sorted = unique_persons.sort_values(by='fullname', ascending=True)    
    logger.debug("Sorted by fullname")

    # Drop the 'box_area' and 'name' column
    unique_persons_sorted = unique_persons_sorted.drop(columns=['box_area', 'fullname'])  
    
    logger.info(f"Duplicates removed and DataFrame sorted in {(time.time() - start_time):.2f} seconds.")
    return unique_persons_sorted