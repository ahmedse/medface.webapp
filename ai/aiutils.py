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
from load_engines import resnet50_model, senet50_model, vgg16_model, w_detect_face, class_labels, sr_model
register_heif_opener()
from typing import Tuple, Optional
import ast
import inspect
from collections import Counter


def save_df_to_csv_excluding_columns(df, excluded_columns=['face_pixels', 'image_path']):
    excluded_columns=['face_pixels']
    # Get the variable name used for the DataFrame
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    df_name = [var_name for var_name, var_val in callers_local_vars if var_val is df]
    
    if not df_name:
        raise ValueError("Could not find the DataFrame variable name.")

    # Define the CSV file name based on the DataFrame variable name
    file_name = f"{df_name[0]}.csv"

    # Select all columns except the excluded ones
    columns_to_save = [column for column in df.columns if column not in excluded_columns]

    # Save the DataFrame to a CSV file without the excluded columns
    df.to_csv(file_name, index=False, columns=columns_to_save)
    print(f"DataFrame saved to {file_name}, excluding columns: {', '.join(excluded_columns)}.")

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


def enhance_face2(face_pixels):
    logger.info("Starting enhancement process for face image.")
    # Check if the image is too small
    if face_pixels.shape[0] < 224 or face_pixels.shape[1] < 224:
        logger.info("Image is too small, applying super-resolution.")
        # Upsample the image using the model
        face_pixels = sr_model.upsample(face_pixels)
        # logger.info("Super-resolution applied.")
        # Resize to an intermediate size larger than the target to downscale later for better quality
        face_pixels = cv2.resize(face_pixels, None, fx=(224/face_pixels.shape[1]), fy=(224/face_pixels.shape[0]), interpolation=cv2.INTER_CUBIC)
        # logger.info("Image resized to intermediate dimensions.")

    # Denoise the image
    face_pixels = cv2.fastNlMeansDenoisingColored(face_pixels, None, 10, 10, 7, 21)
    # logger.info("Image denoising applied.")
    # Improve the contrast (Histogram Equalization)
    # Convert to YUV color space
    img_yuv = cv2.cvtColor(face_pixels, cv2.COLOR_BGR2YUV)
    # logger.info("Image converted to YUV color space.")
    # Apply histogram equalization only on the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # logger.info("Histogram equalization applied on Y channel.")
    # Convert back to BGR color space
    face_pixels = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    # logger.info("Image converted back to BGR color space.")
    # Resize to the target size (224x224)
    final_face_pixels = cv2.resize(face_pixels, (224, 224), interpolation=cv2.INTER_AREA)
    # logger.info("Image resized to target dimensions.")
    # Print image stats
    # logger.info("Image shape: {}".format(final_face_pixels.shape))
    # logger.info("Image dtype: {}".format(final_face_pixels.dtype))
    # logger.info("Max pixel value: {}".format(np.max(final_face_pixels)))
    # logger.info("Min pixel value: {}".format(np.min(final_face_pixels)))
    # logger.info("Enhancement process completed.")
    return final_face_pixels

def enhance_face(face_pixels, save_dir='tmp'):
    logger.info("Starting enhancement process for face image.")

    # Check if the save directory exists, if not, create it
    os.makedirs(save_dir, exist_ok=True)

    # Enhancing the image
    if face_pixels.shape[0] < 100 or face_pixels.shape[1] < 100:
        # Assuming sr_model is a pre-defined super-resolution model
        face_pixels = sr_model.upsample(face_pixels)  # Upsample the image
        face_pixels = cv2.resize(face_pixels, (200, 200), interpolation=cv2.INTER_CUBIC)

        face_pixels = cv2.fastNlMeansDenoisingColored(face_pixels, None, 10, 10, 7, 21)
        img_yuv = cv2.cvtColor(face_pixels, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        final_face_pixels = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        # final_face_pixels = cv2.resize(final_face_pixels, (224, 224), interpolation=cv2.INTER_AREA)
    else:
        final_face_pixels= face_pixels

    # Generate a unique filename using the current timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_filename = f"enhanced_face_{timestamp}.png"
    save_path = os.path.join(save_dir, unique_filename)

    # Save the enhanced image
    cv2.imwrite(save_path, final_face_pixels)
    logger.info(f"Enhanced image saved at {save_path}.")

    return final_face_pixels
    
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
            # enhance face
            face_pixels= enhance_face(face_pixels)
            logger.debug(f"Enhancing face with size {width, height}.")
            # face_check = w_detect_face(face_pixels)
            if True: # face_check disabled
                detected_faces.append({
                    'face_pixels': face_pixels,            
                    'box': box,
                    'confidence': confidence,
                })
    logger.debug(f"Detection took {(time.time() - start_time):.2f} seconds. Detected {len(detected_faces)} faces.")
    return detected_faces


def extract_faces(medsession_dir, min_size=(30, 30), confidence_threshold=0.6):
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
    save_df_to_csv_excluding_columns(faces_df)
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

def query_models(faces_df):
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
        predictions = predict_in_batches(face_pixels, model, batch_size=16)

        logger.debug("Updating DataFrame with predictions...")
        class_confidences = np.max(predictions, axis=1)
        class_indices = np.argmax(predictions, axis=1)

        person_df[model_name + '_label'] = [class_labels[i] for i in class_indices]
        person_df[model_name + '_confidence'] = class_confidences

        logger.debug(f"Made predictions with {model_name} model.")
    
    # Assuming the save_df_to_csv_excluding_columns function has been defined elsewhere and is available
    save_df_to_csv_excluding_columns(person_df)
    
    return person_df

def elect_answer_old(person_df, weights=[0.1, 0.3, 0.5], confidence_threshold=0.95):
    start_time = time.time()
    elected_df = person_df.copy()
    
    # New columns for elected label and confidence
    elected_df['elected_label'] = ''
    elected_df['elected_confidence'] = 0.0
    
    # Check if the number of weights matches the number of models
    if len(weights) != 3:
        raise ValueError("The number of weights must match the number of models (3).")
    
    for i, person in elected_df.iterrows():
        # Filter out models with confidence below the threshold and prepare lists for the ones above the threshold
        valid_labels = []
        valid_confidences = []
        valid_weights = []
        for model_weight, model in zip(weights, ['vgg16', 'resnet50', 'senet50']):
            if person[f'{model}_confidence'] >= confidence_threshold:
                valid_labels.append(person[f'{model}_label'])
                valid_confidences.append(person[f'{model}_confidence'])
                valid_weights.append(model_weight)
        
        # Skip this person if all models are below the confidence threshold
        if not valid_labels:
            logger.debug(f"No models met the confidence threshold for row {i}.")
            continue
        
        # Majority voting mechanism
        label_counts = Counter(valid_labels)
        majority_vote_label, _ = label_counts.most_common(1)[0]
        majority_vote_confidences = [conf for label, conf in zip(valid_labels, valid_confidences) if label == majority_vote_label]
        majority_vote_confidence = np.mean(majority_vote_confidences)
        
        # Weighted voting mechanism
        valid_weights_np = np.array(valid_weights)
        valid_confidences_np = np.array(valid_confidences)
        weighted_confidences = valid_confidences_np * valid_weights_np
        weighted_vote_label = valid_labels[np.argmax(weighted_confidences)]
        weighted_vote_confidence = valid_confidences_np[np.argmax(weighted_confidences)]
        
        # Choose the label with the highest confidence from either majority or weighted voting
        final_decision_label = majority_vote_label if majority_vote_confidence > weighted_vote_confidence else weighted_vote_label
        final_decision_confidence = max(majority_vote_confidence, weighted_vote_confidence)

        # Assign label and confidence to DataFrame
        elected_df.at[i, 'elected_label'] = final_decision_label
        elected_df.at[i, 'elected_confidence'] = final_decision_confidence

        logger.debug(f"Elected label for row {i}: {final_decision_label} with confidence {final_decision_confidence}")

    logger.info(f"Elected answers for {len(elected_df)} rows in {(time.time() - start_time):.2f} seconds.")

    # Assuming the save_df_to_csv_excluding_columns function has been defined elsewhere and is available
    save_df_to_csv_excluding_columns(elected_df)
    
    return elected_df

def elect_answer1(person_df, weights=[0.3, 0.5], confidence_threshold=0.95):
    start_time = time.time()
    elected_df = person_df.copy()
    
    # New columns for elected label and confidence
    elected_df['elected_label'] = ''
    elected_df['elected_confidence'] = 0.0
    
    # Check if the number of weights matches the number of models
    if len(weights) != 2:
        raise ValueError("The number of weights must match the number of models (2).")
    
    for i, person in elected_df.iterrows():
        # Filter out models with confidence below the threshold and prepare lists for the ones above the threshold
        valid_labels = []
        valid_confidences = []
        valid_weights = []
        for model_weight, model in zip(weights, ['resnet50', 'senet50']):
            if person[f'{model}_confidence'] >= confidence_threshold:
                valid_labels.append(person[f'{model}_label'])
                valid_confidences.append(person[f'{model}_confidence'])
                valid_weights.append(model_weight)
        
        # Skip this person if all models are below the confidence threshold
        if not valid_labels:
            logger.debug(f"No models met the confidence threshold for row {i}.")
            continue
        
        # Majority voting mechanism
        label_counts = Counter(valid_labels)
        majority_vote_label, _ = label_counts.most_common(1)[0]
        majority_vote_confidences = [conf for label, conf in zip(valid_labels, valid_confidences) if label == majority_vote_label]
        majority_vote_confidence = np.mean(majority_vote_confidences)
        
        # Weighted voting mechanism
        valid_weights_np = np.array(valid_weights)
        valid_confidences_np = np.array(valid_confidences)
        weighted_confidences = valid_confidences_np * valid_weights_np
        weighted_vote_label = valid_labels[np.argmax(weighted_confidences)]
        weighted_vote_confidence = valid_confidences_np[np.argmax(weighted_confidences)]
        
        # Choose the label with the highest confidence from either majority or weighted voting
        final_decision_label = majority_vote_label if majority_vote_confidence > weighted_vote_confidence else weighted_vote_label
        final_decision_confidence = max(majority_vote_confidence, weighted_vote_confidence)

        # Assign label and confidence to DataFrame
        elected_df.at[i, 'elected_label'] = final_decision_label
        elected_df.at[i, 'elected_confidence'] = final_decision_confidence

        logger.debug(f"Elected label for row {i}: {final_decision_label} with confidence {final_decision_confidence}")

    logger.info(f"Elected answers for {len(elected_df)} rows in {(time.time() - start_time):.2f} seconds.")

    # Assuming the save_df_to_csv_excluding_columns function has been defined elsewhere and is available
    save_df_to_csv_excluding_columns(elected_df)
    
    return elected_df

def elect_answer(person_df, confidence_threshold=0.75, csv_filename='elected_labels.csv'):
    elected_df = person_df.copy()

    # New columns for elected label and confidence
    elected_df['elected_label'] = ''
    elected_df['elected_confidence'] = 0.0

    for i, person in elected_df.iterrows():
        # Collect all labels and confidences
        labels_and_confidences = [
            (person['resnet50_label'], person['resnet50_confidence']),
            (person['senet50_label'], person['senet50_confidence']),
            # (person['vgg16_label'], person['vgg16_confidence'])
             
        ]

        # Sort by confidence in descending order to easily select the highest confidence
        labels_and_confidences.sort(key=lambda x: x[1], reverse=True)

        # Check for a majority label
        label_counts = Counter([label for label, _ in labels_and_confidences])
        majority_vote_label, majority_count = label_counts.most_common(1)[0]

        # If there is a majority
        if majority_count > 1:
            majority_confidences = [conf for label, conf in labels_and_confidences if label == majority_vote_label]
            elected_label = majority_vote_label
            elected_confidence = np.mean(majority_confidences)
        else:
            # No majority, so choose the label with the highest confidence
            elected_label, elected_confidence = labels_and_confidences[0]

        # Assign the label and confidence to the DataFrame: senet50, resnet50, vgg16
        elected_df.at[i, 'elected_label'] = person['resnet50_label']# elected_label
        elected_df.at[i, 'elected_confidence'] = person['resnet50_confidence'] #elected_confidence

    # Save the DataFrame to a CSV file
    elected_df.to_csv(csv_filename, index=False)

    # Filter the DataFrame based on the confidence threshold
    filtered_df = elected_df[elected_df['elected_confidence'] >= confidence_threshold]

    return filtered_df

def remove_duplicates(elected):
    start_time = time.time()
    logger.info(f"Total records before removing duplicates: {len(elected)}")

    # Assuming 'elected_label' and 'elected_confidence' columns exist,
    # we group by 'elected_label', and then use idxmax to find the index of the row with the maximum 'elected_confidence' for each group.
    idx = elected.groupby('elected_label')['elected_confidence'].idxmax()
    unique_persons = elected.loc[idx]

    logger.info(f"Total records after removing duplicates: {len(unique_persons)}")
    logger.info(f"Duplicates removed in {(time.time() - start_time):.2f} seconds.")

    # Assuming the save_df_to_csv_excluding_columns function has been defined elsewhere and is available
    save_df_to_csv_excluding_columns(unique_persons)

    return unique_persons