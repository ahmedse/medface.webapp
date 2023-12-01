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
    default='ERROR' # INFO, ERRORS, DEBUG
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


def reprocess_medsession(medsession_id):           
    try:
        start_time = time.time()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"medsession_id: {medsession_id}") 

        medsession = u.get_object_or_404(Medsession, sessionid=medsession_id)
        task, created = TaskStatus.objects.get_or_create(medsession=medsession)

        report_task_status(task, 'INIT', 'Initializing', 0, start_time)
        
        medsession_dir = u.get_medsession_images_dir(medsession)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Medsession directory: {medsession_dir}")        

        if not os.path.exists(medsession_dir):
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"Medsession directory does not exist: {medsession_dir}")
            return True

        report_task_status(task, 'PROG', 'Faces detection: Extract faces', 5, start_time)
        
        faces_df = ai.extract_faces(medsession_dir)
        num_faces = len(faces_df)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Extracted faces: {num_faces}") 

        report_task_status(task, 'PROG', f'Faces detection: Extracted {num_faces} faces', 25, start_time)  

        report_task_status(task, 'PROG', f'Person recognition: Recognizing persons', 30, start_time)               
        person_df = ai.query_models(faces_df)
        num_persons = len(person_df)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Queried models: {num_persons}")        

        report_task_status(task, 'PROG', f'Person recognition: Recognized {num_persons} persons', 50, start_time)

        report_task_status(task, 'PROG', f'Sorting: AI magic started', 55, start_time)
        elected = ai.elect_answer(person_df)
        num_elected = len(elected)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Elected answers: {num_elected}")        

        report_task_status(task, 'PROG', f'Sorting: Keep doing magic', 75, start_time)

        unique_persons = ai.remove_duplicates(elected)
        num_unique_persons = len(unique_persons)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Unique persons: {num_unique_persons}")        

        report_task_status(task, 'PROG', f'Finishing: AI saw {num_persons} persons', 90, start_time)

        update_medsessionpersons(unique_persons, medsession)
        
        report_task_status(task, 'PROG', f'Finishing: Saved in database', 99, start_time)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Medsession {medsession_id} processed successfully.")

        report_task_status(task, 'DONE', f'completed', 100, start_time)
        total_time = time.time() - start_time
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Total processing time: {total_time:.2f} seconds.")
        return True
    except Http404:
        if logger.isEnabledFor(logging.ERROR):
            logger.error(f"Medsession with id {medsession_id} not found.")
        report_task_status(task, 'FAIL', f'[ERROR] Medsession with id {medsession_id} not found.', 0, start_time)
    except Exception as e:
        if logger.isEnabledFor(logging.ERROR):
            logger.error(f"An unexpected error occurred while processing medsession {medsession_id}: {str(e)}", exc_info=True)
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
        time.sleep(.1)  # sleep just 1 second        