import sys
from pathlib import Path
# add the top level directory to the sys.path
project_directory = Path(__file__).resolve().parent.parent
sys.path.append(str(project_directory))
import django
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'webapp.settings'
django.setup()
import redis
import time
import json
import webapp.utils as u
import aiutils as ai
from medapp.models import Medsession, MedsessionPerson, TaskStatus
from django.db import transaction
import traceback
from django.shortcuts import get_object_or_404
from django.http import Http404
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def report_task_status(task_status, status, progress):
    task_status.status = status
    task_status.progress = progress  # 25% done
    task_status.save()

def reprocess_medsession(medsession_id):           
    try:
        logger.debug(f"medsession_id: {medsession_id}") 
        print(f"medsession_id: {medsession_id}")
        medsession = u.get_object_or_404(Medsession, sessionid=medsession_id)
        task_status, created = TaskStatus.objects.get_or_create(medsession=medsession)
        print(f"task_status  created: {task_status}")
        # Update the task status
        report_task_status(task_status, 'Initializing', 0)
        
        medsession_dir = u.get_medsession_images_dir(medsession)
        logger.info(f"Medsession directory: {medsession_dir}")        
        if not os.path.exists(medsession_dir):
            logger.info(f"Medsession directory does not exist: {medsession_dir}")
            return True
        with transaction.atomic():
            # Delete medsession_persons related to this medsession
            deleted_count = MedsessionPerson.objects.filter(medsession=medsession).delete()
            logger.info(f"Deleted {deleted_count} MedsessionPerson records.")   
        report_task_status(task_status, 'Extract faces', 5)
        # Perform face recognition and create MedsessionPerson objects                
        faces_df = ai.extract_faces(medsession_dir)
        logger.info(f"Extracted faces: {len(faces_df)}") 
        report_task_status(task_status, f'Extracted {len(faces_df)} faces', 25)  

        report_task_status(task_status, f'Recognizing persons', 30)       
        person_df = ai.query_models(faces_df)
        logger.info(f"Queried models: {len(person_df)}")        
        report_task_status(task_status, f'Recognized {len(person_df)} persons', 50)

        report_task_status(task_status, f'AI magic started', 55)
        elected = ai.elect_answer(person_df)
        logger.info(f"Elected answers: {len(elected)}")        
        report_task_status(task_status, f'Keep doing magic', 75)

        unique_persons = ai.remove_duplicates(elected)
        logger.info(f"Unique persons: {len(unique_persons)}")        
        report_task_status(task_status, f'AI saw {len(person_df)} persons', 90)

        medsession_persons = u.create_medsession_persons(unique_persons, medsession)
        report_task_status(task_status, f'Saved in database', 99)
        logger.info(f"Created MedsessionPerson records: {len(medsession_persons)}")        
        # If everything is successful, log a success message
        logger.info(f"Medsession {medsession_id} processed successfully.")
        report_task_status(task_status, f'completed', 100)
        return True
    except Http404:
        logger.error(f"Medsession with id {medsession_id} not found.")
        report_task_status(task_status, f'[ERROR] Medsession with id {medsession_id} not found.', 0)
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing medsession {medsession_id}: {str(e)}", exc_info=True)
        print(f"An unexpected error occurred while processing medsession {medsession_id}: {str(e)}")
        report_task_status(task_status, f'[ERROR] An unexpected error occurred while processing medsession {medsession_id}: {str(e)}.', 0)

def process_images(message):
    logger.debug(f"Received message: {message}")
    if message['type'] == 'message':
        try:
            data = json.loads(message['data'].decode('utf-8'))  # add .decode('utf-8') here
            medsession_id = data['medsession_id']
            reprocess_medsession(medsession_id)
        except Exception as e:
            logger.error(f"An error occurred while processing message {message}: {str(e)}", exc_info=True)
    else:
        logger.debug(f"Received non-message: {message['type']}")

if __name__ == "__main__":    
    r = redis.Redis(host='localhost', port=6379, db=0)    
    p = r.pubsub()
    p.subscribe(**{'medsession-channel': process_images})
    logger.info(f"p.subscribe done")
    print(f"Listening for messages...")
    logger.info(f"Listening for messages...")
    while True:
        try:
            message = p.get_message()
            if message:
                print(f"Got message: {message}")
                process_images(message)
        except Exception as e:
            logger.error(f"An error occurred while getting message: {str(e)}", exc_info=True)
        time.sleep(1)  # sleep just 1 second