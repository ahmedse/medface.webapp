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
from datetime import datetime, timedelta
import time
import json
import webapp.utils as u
import aiutils as ai
from medapp.models import Medsession, MedsessionPerson, TaskStatus, Person
from django.db import transaction
import traceback
from django.shortcuts import get_object_or_404
from django.http import Http404
import argparse
import signal
import sys
import logging
import concurrent.futures
# from tenacity import retry, stop_after_attempt, wait_exponential

# setup argument parsing
parser = argparse.ArgumentParser(description="redis-worker Service")
parser.add_argument(
    '--loglevel', 
    help='Set log level', 
    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
    default='DEBUG' # INFO
)
args = parser.parse_args()

# Configure logging
loglevel = getattr(logging, args.loglevel)
logging.basicConfig(filename='redisworker_service.log', level=loglevel) # /var/log/medface/
logger = logging.getLogger(__name__)

def report_task_status(task, status, status_message, progress, start_time):
    # Check if status is a valid choice
    valid_status_choices = [choice[0] for choice in TaskStatus.STATUS_CHOICES]
    if status not in valid_status_choices:
        raise ValueError(f"Invalid status. Expected one of: {valid_status_choices}")

    # Update task object
    task.status = status
    task.msg = status_message
    task.progress = progress
    task.TimeTaken = time.time() - start_time
    task.save()

def update_medsessionpersons(unique_persons, medsession):
    # Get all MedsessionPerson instances associated with the Medsession
    medsessionpersons = MedsessionPerson.objects.filter(medsession=medsession)

    # Convert unique_persons DataFrame elected_label column to a set for quick lookup
    unique_persons_set = set(unique_persons['elected_label'])

    # Update existing MedsessionPerson instances
    for medsessionperson in medsessionpersons:
        person_label = f"{medsessionperson.person.regno}_{medsessionperson.person.fullname}"
        if person_label in unique_persons_set:
            # Update MedsessionPerson with new data
            person_data = unique_persons[unique_persons['elected_label'] == person_label].iloc[0]
            medsessionperson.box = str(person_data['box'])
            medsessionperson.confidence = person_data['confidence']
            medsessionperson.image_path = person_data['image_path']
            medsessionperson.resnet50_label = person_data['resnet50_label']
            medsessionperson.resnet50_confidence = person_data['resnet50_confidence']
            medsessionperson.senet50_label = person_data['senet50_label']
            medsessionperson.senet50_confidence = person_data['senet50_confidence']
            medsessionperson.vgg16_label = person_data['vgg16_label']
            medsessionperson.vgg16_confidence = person_data['vgg16_confidence']
            medsessionperson.elected_label = person_data['elected_label']
            medsessionperson.status = '2'  # Update status to 'AI'
            medsessionperson.attendance = '1'  
            medsessionperson.save()
            # Remove person_label from the set after updating MedsessionPerson
            unique_persons_set.remove(person_label)
        else:
            # Update status to 'Not Detected'
            medsessionperson.status = '5'
            medsessionperson.save()

    # # Create new MedsessionPerson instances for the remaining unique_persons
    # for person_label in unique_persons_set:
    #     person_data = unique_persons[unique_persons['elected_label'] == person_label].iloc[0]
    #     regno, fullname = person_label.split('_')
    #     person_obj = Person.objects.get(regno=regno, fullname=fullname)
    #     MedsessionPerson.objects.create(
    #         medsession=medsession,
    #         person=person_obj,
    #         box=str(person_data['box']),
    #         confidence=person_data['confidence'],
    #         image_path=person_data['image_path'],            
    #         resnet50_label=person_data['resnet50_label'],
    #         resnet50_confidence=person_data['resnet50_confidence'],
    #         senet50_label=person_data['senet50_label'],
    #         senet50_confidence=person_data['senet50_confidence'],
    #         vgg16_label=person_data['vgg16_label'],
    #         vgg16_confidence=person_data['vgg16_confidence'],
    #         elected_label=person_data['elected_label'],
    #         attendance = '1', # present
    #         status='2',  # Set status to 'AI' for newly created MedsessionPersons
    #     )
   
def reprocess_medsession(medsession_id):           
    try:
        start_time = time.time()
        logger.debug(f"medsession_id: {medsession_id}") 
        print(f"medsession_id: {medsession_id}")
        medsession = u.get_object_or_404(Medsession, sessionid=medsession_id)
        task, created = TaskStatus.objects.get_or_create(medsession=medsession)
        print(f"task_status  created: {task}")
        # Update the task status
        report_task_status(task, 'INIT', 'Initializing', 0, start_time)
        
        medsession_dir = u.get_medsession_images_dir(medsession)
        logger.info(f"Medsession directory: {medsession_dir}")        
        if not os.path.exists(medsession_dir):
            logger.info(f"Medsession directory does not exist: {medsession_dir}")
            return True
        # with transaction.atomic():
        #     # Delete medsession_persons related to this medsession
        #     deleted_count = MedsessionPerson.objects.filter(medsession=medsession).delete()
        #     logger.info(f"Deleted {deleted_count} MedsessionPerson records.")   

        report_task_status(task, 'PROG', 'Faces detection: Extract faces', 5, start_time)
        
        # Perform face recognition and create MedsessionPerson objects                
        faces_df = ai.extract_faces(medsession_dir)
        logger.info(f"Extracted faces: {len(faces_df)}") 
        print(f"[DEBUG] Extracted faces: {len(faces_df)}")
        report_task_status(task, 'PROG', f'Faces detection: Extracted {len(faces_df)} faces', 25, start_time)  

        report_task_status(task, 'PROG', f'Person recognition: Recognizing persons', 30, start_time)               
        person_df = ai.query_models(faces_df)
        print(f"[DEBUG] Queried models: {len(person_df)}")
        logger.info(f"Queried models: {len(person_df)}")        
        report_task_status(task, 'PROG', f'Person recognition: Recognized {len(person_df)} persons', 50, start_time)

        report_task_status(task, 'PROG', f'Sorting: AI magic started', 55, start_time)
        elected = ai.elect_answer(person_df)
        print(f"[DEBUG] Elected answers: {len(elected)}")
        logger.info(f"Elected answers: {len(elected)}")        
        report_task_status(task, 'PROG', f'Sorting: Keep doing magic', 75, start_time)

        # unique_persons = handle_timeout(ai.remove_duplicates, timeout_seconds, elected)
        unique_persons = ai.remove_duplicates(elected)
        print(f"[DEBUG] Unique persons: {len(unique_persons)}")
        logger.info(f"Unique persons: {len(unique_persons)}")        
        report_task_status(task, 'PROG', f'Finishing: AI saw {len(person_df)} persons', 90, start_time)

        update_medsessionpersons(unique_persons, medsession)
        
        report_task_status(task, 'PROG', f'Finishing: Saved in database', 99, start_time)
        # logger.info(f"Created MedsessionPerson records: {len(medsession_persons)}")        
        # If everything is successful, log a success message
        print(f"[DEBUG] Medsession {medsession_id} processed successfully.")
        logger.info(f"Medsession {medsession_id} processed successfully.")
        report_task_status(task, 'DONE', f'completed', 100, start_time)
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds.")
        logger.info(f"Total processing time: {total_time:.2f} seconds.")
        return True
    except Http404:
        print(f"[DEBUG] Medsession with id {medsession_id} not found.")
        logger.error(f"Medsession with id {medsession_id} not found.")
        report_task_status(task, 'FAIL', f'[ERROR] Medsession with id {medsession_id} not found.', 0, start_time)
    except Exception as e:
        print(f"[DEBUG] An unexpected error occurred while processing medsession {medsession_id}: {str(e)}.")
        logger.error(f"An unexpected error occurred while processing medsession {medsession_id}: {str(e)}", exc_info=True)
        print(f"An unexpected error occurred while processing medsession {medsession_id}: {str(e)}")
        report_task_status(task, 'FAIL', f'[ERROR] An unexpected error occurred while processing medsession {medsession_id}: {str(e)}.', 0, start_time)

def cleanup_tasks():
    ten_minutes_ago = datetime.now() - timedelta(minutes=10)
    two_minute_ago = datetime.now() - timedelta(minutes=2)
    # Only consider tasks that have been marked as read
    stuck_tasks = TaskStatus.objects.filter(last_updated__lte=ten_minutes_ago, status__in=['INIT', 'PROG'], read=True)
    done_tasks = TaskStatus.objects.filter(last_updated__lte=two_minute_ago, status='DONE', read=True)
    failed_tasks = TaskStatus.objects.filter(last_updated__lte=two_minute_ago, status='FAIL', read=True)

    for task in stuck_tasks | done_tasks | failed_tasks:
        # Perform cleanup
        task.delete()
        print("[DEBUG] Running cleanup tasks: {task}")

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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
    last_cleanup = datetime.now()
    while True:
        try:
            message = p.get_message()
            if message:
                print(f"Got message: {message}")
                process_images(message)
            # Cleanup every 10 seconds
            if datetime.now() - last_cleanup > timedelta(seconds= 60):
                cleanup_tasks()
                last_cleanup = datetime.now()
        except Exception as e:
            logger.error(f"An error occurred while getting message: {str(e)}", exc_info=True)
        time.sleep(1)  # sleep just 1 second        