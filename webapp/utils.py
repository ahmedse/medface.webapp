import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from PIL import Image
import logging
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
from PIL import Image
from PIL.ExifTags import TAGS
from typing import Tuple, Optional
import ast
import tempfile
from medapp.models import MedsessionPerson
from django.conf import settings
from django.http import Http404
from django.shortcuts import get_object_or_404
import redis
# Create a global Redis connection
r = redis.Redis(host='localhost', port=6379, db=0)

def get_image_url(image_path):    
    """Return the URL of an image given its file path."""
    image_url = settings.MEDIA_URL + os.path.relpath(image_path, settings.MEDIA_ROOT).replace("\\", "/") 
    return image_url

def get_medsession_images_dir(medsession):
    """Return the directory where the images for a given medsession are stored."""
    images_dir = os.path.join(
        settings.MEDIA_ROOT, 
        r'runs', 
        str(medsession.year), 
        str(medsession.term), 
        str(medsession.day), 
        str(medsession.period), 
        str(medsession.room)
    )
    return images_dir

def get_medsession_images(medsession):    
    # Calculate medsession path
    medsession_dir = get_medsession_images_dir(medsession)
    # List to hold all image URLs and paths
    image_data = []
    try:
        # List all files in the directory        
        for filename in os.listdir(medsession_dir):
            # Construct the file path
            file_path = os.path.join(medsession_dir, filename)
            # Only add jpg and png files
            if os.path.isfile(file_path) and file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Construct the URL
                file_url = get_image_url(file_path)              
                # Construct the image path as sessionid/filename
                image_path = f"{medsession.sessionid}_{filename}"                    
                # Add the data to the list
                image_data.append({
                    'url': file_url,
                    'image_path': image_path
                })                
                
    except FileNotFoundError:
        # Handle the error if the medsession directory does not exist
        raise Http404("Medsession directory does not exist.")
    except PermissionError:
        # Handle the error if there are insufficient permissions to read the directory or a file
        raise Http404("Insufficient permissions to read the medsession directory or a file.")
    except Exception as e:
        # Handle all other exceptions
        logger.error("An error occurred while reading the medsession directory: %s", str(e))
        raise Http404("An error occurred while reading the medsession directory.")
    # Return the list of image URLs and paths
    return image_data

def get_label(person):
    """Return the corrected_label if it exists, otherwise return the elected_label."""
    return person.corrected_label if person.corrected_label else person.elected_label

def get_personal_photo_url(label):
    """Return the URL of the personal photo for a given label, or None if no such photo exists."""
    personal_photo_dir = os.path.join(settings.MEDIA_ROOT, 'y3.personal', label)
    try:
        personal_photo_files = os.listdir(personal_photo_dir)
    except FileNotFoundError:
        personal_photo_files = []

    if personal_photo_files:
        personal_photo_path = os.path.join(personal_photo_dir, personal_photo_files[0])
        personal_photo_url = get_image_url(personal_photo_path)
    else:
        personal_photo_url = None

    return personal_photo_url

def get_person_image_url(medsession, label, image_path, box):
    """Return the URL of the person image, processing and saving the image if necessary."""
    temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", prefix=f"{label}_", dir=temp_dir) as temp_file:
        save_path = temp_file.name

    src_face_path = os.path.join(get_medsession_images_dir(medsession), image_path)
    save_face_from_src(src_face_path, box, save_path)

    return get_image_url(save_path)


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