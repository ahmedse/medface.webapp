import os
import warnings
warnings.filterwarnings('ignore')
from keras_vggface.utils import preprocess_input
import requests
import numpy as np
import pandas as pd
import cv2
from django.core.exceptions import ValidationError
from keras_vggface.utils import preprocess_input
from PIL import Image
import logging
logger = logging.getLogger(__name__)
from PIL import ExifTags
from pillow_heif import register_heif_opener
from .ai import resnet50_model, senet50_model, vgg16_model, w_detect_face, class_labels
register_heif_opener()

import ast

def process_image(image_path, box, save_path, size=(300, 500)):
    # print(f"[DEBUG] src image: {image_path}")
    # print(f"[DEBUG] save extracted face: {save_path}")

    # Check if the image file exists
    if not os.path.isfile(image_path):
        logging.error(f"Image file does not exist: {image_path}")
        return None

    try:
        # Open image and convert it to RGB
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Failed to open and convert image: {e}")
        return None

    logging.info(f"Successfully loaded image: {image_path}")

    # Parse the box string into a list of integers
    try:
        box = ast.literal_eval(box)
    except ValueError as e:
        logging.error(f"Failed to parse box string: {e}")
        return None

    logging.info(f"Parsed box: {box}")

    # Calculate box coordinates with margin
    try:
        box = [box[0]-10, box[1]-10, box[0]+box[2]+10, box[1]+box[3]+10]
    except Exception as e:
        logging.error(f"Failed to calculate box coordinates: {e}")
        return None

    logging.info(f"Calculated box coordinates: {box}")

    # Crop image
    try:
        img_cropped = img.crop(box)
    except Exception as e:
        logging.error(f"Failed to crop image: {e}")
        return None

    logging.info(f"Cropped image size: {img_cropped.size}")

    # Resize image
    img_resized = img_cropped.resize(size)
    logging.info(f"Resized image size: {img_resized.size}")

    try:
        img_resized.save(save_path)
    except Exception as e:
        logging.error(f"Failed to save image: {e}")
        return None

    logging.info(f"Successfully processed and saved image: {save_path}")
    return img_resized

from PIL import Image
from PIL.ExifTags import TAGS

def correct_image_orientation(image_path):
    try:   
        with Image.open(image_path) as image:
            exif = None
            if hasattr(image, "_getexif") and image._getexif() is not None:
                exif = {TAGS[key]: value for key, value in image._getexif().items() if key in TAGS and isinstance(TAGS[key], str)}
                print(f'[DEBUG] exif: {exif}')
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

def extract_faces(medsession_dir, min_size=(50, 50)):
    
    faces_df = []
      
    for file_name in os.listdir(medsession_dir):
        file_path = os.path.join(medsession_dir, file_name)        

        if not os.path.exists(file_path):
            raise ValidationError("File path does not exist.")

        if os.path.getsize(file_path) == 0:
            raise ValidationError("File is empty.")
        
        try:
            img = correct_image_orientation(file_path)            
        except IOError:
            raise ValidationError(f"File {file_name} is not a valid image.")

        img = np.array(img, dtype='uint8')
        img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        faces = w_detect_face(img_cv2)
        print(f'[DEBUG] image: {file_path}, size: {len(np.array(img))}')

        if not faces:
            print(f'[DEBUG] No faces detected in the image: {file_path}.')
            continue
        
        for face in faces:
            box, confidence = face['box'], face['confidence']
            x, y, width, height = box

            # Skip faces that are too small
            if width < min_size[0] or height < min_size[1]:
                continue
            
            # Extract face pixels
            face_pixels = img_cv2[y: y + height, x: x + width]
            
            # Check again if the extracted face is really a face
            face_check = w_detect_face(face_pixels)       
            if not face_check:
                continue
            
            faces_df.append({    
                'face_pixels': face_pixels,            
                'box': box,
                'confidence': confidence,
                'image_path': file_name,
            })
        
        print(f'[DEBUG] {len(faces_df)} faces extracted from image: {file_path}.')
        
    faces_df = pd.DataFrame(faces_df)
    return faces_df

def predict_in_batches(face_pixels, model, batch_size=64):
    predictions = []
    for i in range(0, face_pixels.shape[0], batch_size):
        batch = face_pixels[i:i+batch_size]
        batch_predictions = model.predict(batch)
        predictions.extend(batch_predictions)
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
        print(f"[DEBUG] Making predictions with {model_name} model...")
        predictions = predict_in_batches(face_pixels, model, batch_size=32)

        print(f"[DEBUG] Updating DataFrame with predictions...")
        class_indices = np.argmax(predictions, axis=1)
        class_confidences = np.max(predictions, axis=1)

        person_df[model_name + '_label'] = [class_labels[i] for i in class_indices]
        person_df[model_name + '_confidence'] = class_confidences

        # print(f"[DEBUG] Completed predictions with {model_name} model.")
    return person_df

def query_models_by_http(faces_df):
    # Load class labels
    file_path = os.path.join(settings.MEDIA_ROOT, 'face-labels.pickle')
    with open(file_path, 'rb') as f:
        class_labels = pickle.load(f)
    person_df = faces_df.copy()
    person_df.reset_index(drop=True, inplace=True)

    # Initiate columns for each model
    for model in ['resnet50', 'senet50', 'vgg16']:
        person_df[model + '_label'] = ''
        person_df[model + '_confidence'] = 0.0

    for i, face in person_df.iterrows():
        face_pixels = face['face_pixels']
        print(f"[DEBUG] Faces: {len(face_pixels)}. Preprocessing...")
        face_pixels = face_pixels.astype('float64')
        face_pixels = preprocess_input(face_pixels)
        
        print(f"[DEBUG] Start querying the models...")
        # Query each model
        for model in ['resnet50', 'senet50', 'vgg16']:
            print(f"[DEBUG] Querying {model} ...")
            response = requests.post(f'http://localhost:8501/v1/models/{model}:predict', json={"instances": [face_pixels.tolist()]})
            print(f"[DEBUG] Status code: {response.status_code}")
            print(f"[DEBUG] json response:  {response.json()}")
            response_json = response.json()
            if 'predictions' in response_json:
                predictions = response_json['predictions']
                class_index = np.argmax(predictions[0])
                class_confidence = predictions[0][class_index]
                person_df.loc[i, model + '_label'] = class_labels[class_index]
                person_df.loc[i, model + '_confidence'] = class_confidence
                print(f"[DEBUG] Querying {model} Done")
            else:
                print(f"[ERROR] No predictions in response from {model} model")
    return person_df

def elect_answer(person_df):
    # New column for elected label
    person_df['elected_label'] = ''
    
    for i, person in person_df.iterrows():
        # print(f"[DEBUG] Processing row {i}...")
        
        labels = [person['resnet50_label'], person['senet50_label'], person['vgg16_label']]
        confidences = [person['resnet50_confidence'], person['senet50_confidence'], person['vgg16_confidence']]
        
        # Count the occurrences of each label
        label_counts = {label: labels.count(label) for label in labels}
        print(f"[DEBUG] label counts: {label_counts}")
        
        # Find the label(s) with the maximum count
        max_count = max(label_counts.values())
        max_labels = [label for label, count in label_counts.items() if count == max_count]
        print(f"[DEBUG] max labels: {max_labels}")        
        if len(max_labels) == 1:
            # If there's one label with the maximum count, use it
            person_df.loc[i, 'elected_label'] = max_labels[0]
        else:
            # If there's a tie, use the label with the highest average confidence
            avg_confidences = [np.mean([confidences[j] for j in range(3) if labels[j] == label]) for label in max_labels]
            person_df.loc[i, 'elected_label'] = max_labels[np.argmax(avg_confidences)]
        print(f"[DEBUG] Elected label for row {i}: {person_df.loc[i, 'elected_label']}")
    return person_df

def remove_duplicates(elected):
    print("[DEBUG] Total records before removing duplicates: ", len(elected))    
    # Detect duplicates
    duplicates = elected[elected.duplicated(subset='elected_label', keep=False)]
    print("[DEBUG] Duplicates:")
    print(duplicates)    
    
    # Add column for area of bounding box
    elected['box_area'] = elected['box'].apply(lambda b: b[2]*b[3])
    
    # Sort by box_area in descending order so that duplicates with the largest area come first
    elected.sort_values('box_area', ascending=False, inplace=True)
    
    # Remove duplicates, keeping the first one (which, due to the sorting, will be the one with the largest area)
    unique_persons = elected.drop_duplicates(subset='elected_label')
    print("[DEBUG] Total records after removing duplicates: ", len(unique_persons))
    
    # Split 'elected_label' and create a new column 'name'
    unique_persons['fullname'] = unique_persons['elected_label'].apply(lambda x: x.split('_')[1] if '_' in x else x)    
    # Sort DataFrame by 'name' in ascending order
    unique_persons_sorted = unique_persons.sort_values(by='fullname', ascending=True)    
    print("[DEBUG] Sorted by fullname")

    # Drop the 'box_area' and 'name' column
    unique_persons_sorted = unique_persons_sorted.drop(columns=['box_area', 'fullname'])   
    
    return unique_persons_sorted

def remove_duplicates2(elected):
    print("[DEBUG] Total records before removing duplicates: ", len(elected))    
    # Detect duplicates
    duplicates = elected[elected.duplicated(subset='elected_label', keep=False)]
    print("[DEBUG] Duplicates:")
    print(duplicates)    
    # Remove duplicates
    unique_persons = elected.drop_duplicates(subset='elected_label')
    print("[DEBUG] Total records after removing duplicates: ", len(unique_persons))
    # Split 'elected_label' and create a new column 'name'
    unique_persons['fullname'] = unique_persons['elected_label'].apply(lambda x: x.split('_')[1] if '_' in x else x)    
    # Sort DataFrame by 'name' in ascending order
    unique_persons_sorted = unique_persons.sort_values(by='fullname', ascending=True)    
    print("[DEBUG] Sorted by fullname")
    # Drop the 'name' column
    unique_persons_sorted = unique_persons_sorted.drop(columns=['fullname'])    
    return unique_persons_sorted

from medapp.models import MedsessionPerson
def create_medsession_persons(unique_persons, medsession):
    print(unique_persons.info())
    print("[DEBUG] Total persons detected: ", len(unique_persons)) 
    medsession_persons = []
    for _, person in unique_persons.iterrows():
        medsession_person = MedsessionPerson(
            medsession=medsession,
            box=str(person['box']),
            confidence=person['confidence'],
            image_path=person['image_path'],            
            resnet50_label=person['resnet50_label'],
            resnet50_confidence=person['resnet50_confidence'],
            senet50_label=person['senet50_label'],
            senet50_confidence=person['senet50_confidence'],
            vgg16_label=person['vgg16_label'],
            vgg16_confidence=person['vgg16_confidence'],
            elected_label=person['elected_label'],
        )
        # print("[DEBUG] MedsessionPerson object before save: ", medsession_person.__dict__)
        medsession_person.save()
        medsession_persons.append(medsession_person)
    return medsession_persons

# Implement your own functions for face_detection, run_models, make_decision, get_unique, and save_to_db