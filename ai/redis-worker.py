import logging
# Set up logging
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)  # Set the threshold of logger to DEBUG
# Create a file handler for output file
handler = logging.FileHandler('output.log')
# Set the logger output format
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(handler)
# Create a stream handler to print the logs to console
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(consoleHandler)
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


def report_task_status(task_status, phase, status_message, progress_percentage, start_time):
    total_time = time.time() - start_time
    task_status.phase = phase
    task_status.status = status_message
    task_status.progress = progress_percentage
    task_status.TimeTaken = int(total_time)
    task_status.save()

def reprocess_medsession(medsession_id):           
    try:
        start_time = time.time()
        logger.debug(f"medsession_id: {medsession_id}") 
        print(f"medsession_id: {medsession_id}")
        medsession = u.get_object_or_404(Medsession, sessionid=medsession_id)
        task_status, created = TaskStatus.objects.get_or_create(medsession=medsession)
        print(f"task_status  created: {task_status}")
        # Update the task status
        report_task_status(task_status, 'Init', 'Initializing', 0, start_time)
        
        medsession_dir = u.get_medsession_images_dir(medsession)
        logger.info(f"Medsession directory: {medsession_dir}")        
        if not os.path.exists(medsession_dir):
            logger.info(f"Medsession directory does not exist: {medsession_dir}")
            return True
        with transaction.atomic():
            # Delete medsession_persons related to this medsession
            deleted_count = MedsessionPerson.objects.filter(medsession=medsession).delete()
            logger.info(f"Deleted {deleted_count} MedsessionPerson records.")   

        report_task_status(task_status, 'Faces detection', 'Extract faces', 5, start_time)
        
        # Perform face recognition and create MedsessionPerson objects                
        faces_df = ai.extract_faces(medsession_dir)
        logger.info(f"Extracted faces: {len(faces_df)}") 
        report_task_status(task_status, 'Faces detection', f'Extracted {len(faces_df)} faces', 25, start_time)  

        report_task_status(task_status, 'Person recognition', f'Recognizing persons', 30, start_time)       
        person_df = ai.query_models(faces_df)
        logger.info(f"Queried models: {len(person_df)}")        
        report_task_status(task_status, 'Person recognition', f'Recognized {len(person_df)} persons', 50, start_time)

        report_task_status(task_status, 'Sorting', f'AI magic started', 55, start_time)
        elected = ai.elect_answer(person_df)
        logger.info(f"Elected answers: {len(elected)}")        
        report_task_status(task_status, 'Sorting', f'Keep doing magic', 75, start_time)

        unique_persons = ai.remove_duplicates(elected)
        logger.info(f"Unique persons: {len(unique_persons)}")        
        report_task_status(task_status, 'Finishing', f'AI saw {len(person_df)} persons', 90, start_time)

        medsession_persons = u.create_medsession_persons(unique_persons, medsession)
        report_task_status(task_status, 'Finishing', f'Saved in database', 99, start_time)
        logger.info(f"Created MedsessionPerson records: {len(medsession_persons)}")        
        # If everything is successful, log a success message
        logger.info(f"Medsession {medsession_id} processed successfully.")
        report_task_status(task_status, 'Done', f'completed', 100, start_time)
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds.")
        return True
    except Http404:
        logger.error(f"Medsession with id {medsession_id} not found.")
        report_task_status(task_status, 'Failure', f'[ERROR] Medsession with id {medsession_id} not found.', 0, start_time)
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing medsession {medsession_id}: {str(e)}", exc_info=True)
        print(f"An unexpected error occurred while processing medsession {medsession_id}: {str(e)}")
        report_task_status(task_status, 'Failure', f'[ERROR] An unexpected error occurred while processing medsession {medsession_id}: {str(e)}.', 0, start_time)

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