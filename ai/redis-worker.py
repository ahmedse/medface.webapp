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
from medapp.models import Medsession, MedsessionPerson
from django.db import transaction
import traceback
from django.shortcuts import get_object_or_404
from django.http import Http404

def reprocess_medsession(medsession_id):        
    try:
        print(f"[DEBUG] medsession_id: {medsession_id}") 
        medsession = u.get_object_or_404(Medsession, sessionid=medsession_id)
        medsession_dir = u.get_medsession_images_dir(medsession)
        print(f"[INFO] Medsession directory: {medsession_dir}")        
        if not os.path.exists(medsession_dir):
            print(f"[INFO] Medsession directory does not exist: {medsession_dir}")
            return True
        with transaction.atomic():
            # Delete medsession_persons related to this medsession
            deleted_count = MedsessionPerson.objects.filter(medsession=medsession).delete()
            print(f"[INFO] Deleted {deleted_count} MedsessionPerson records.")            
        # Perform face recognition and create MedsessionPerson objects
        faces_df = ai.extract_faces(medsession_dir)
        print(f"[INFO] Extracted faces: {len(faces_df)}")        
        person_df = ai.query_models(faces_df)
        print(f"[INFO] Queried models: {len(person_df)}")        
        elected = ai.elect_answer(person_df)
        print(f"[INFO] Elected answers: {len(elected)}")        
        unique_persons = ai.remove_duplicates(elected)
        print(f"[INFO] Unique persons: {len(unique_persons)}")        
        medsession_persons = u.create_medsession_persons(unique_persons, medsession)
        print(f"[INFO] Created MedsessionPerson records: {len(medsession_persons)}")        
        # If everything is successful, log a success message
        print(f"[SUCCESS] Medsession {medsession_id} processed successfully.")
        return True
    except Http404:
        print(f"[ERROR] Medsession with id {medsession_id} not found.")
    except Exception as e:
        traceback.print_exc()  # This will print the traceback
        print(f"[ERROR] An unexpected error occurred while processing medsession {medsession_id}: {str(e)}")     

def process_images(message):
    print(f"[DEBUG] Received message: {message}")
    if message['type'] == 'message':
        data = json.loads(message['data'].decode('utf-8'))  # add .decode('utf-8') here
        medsession_id = data['medsession_id']
        reprocess_medsession(medsession_id)
    else:
        print(f"[DEBUG] Received non-message: {message['type']}")

if __name__ == "__main__":
    r = redis.Redis(host='localhost', port=6379, db=0)
    p = r.pubsub()
    p.subscribe(**{'medsession-channel': process_images})
    print(f"[INFO] p.subscribe done")

    print(f"[INFO] Listening for messages...")
    while True:
        message = p.get_message()
        if message:
            process_images(message)
        time.sleep(1)  # sleep just 1 second