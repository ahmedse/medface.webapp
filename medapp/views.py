
# Create your views here.
import os
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
from django.shortcuts import render, redirect, get_object_or_404
from .forms import MedsessionForm, PersonForm, ImageForm
from .models import Medsession, Person, MedsessionPerson, TaskStatus, MedsessionGroup
from django.urls import reverse_lazy
from django.views.generic import ListView, UpdateView, DeleteView
from .models import Image
from .filters import MedsessionFilter
from django.contrib import messages
from django.forms import modelformset_factory
from django.core.exceptions import ValidationError
from django.conf import settings
import tempfile
import webapp.utils as u
import logging
from django.http import FileResponse
from xhtml2pdf import pisa
from django.template.loader import get_template
from io import BytesIO
from django.contrib.staticfiles import finders
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import io
from django.views.decorators.http import require_POST
from django.http import Http404
from django.db import transaction
import traceback
from django.db.models import Q
from django.shortcuts import redirect
import redis
import json
from django.urls import reverse_lazy
from django.views.generic import UpdateView
from django.db.models import Count
import time
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect, get_object_or_404
from .models import Medsession, MedsessionPerson
from .forms import MedsessionForm
from django.contrib import messages
from django.db import IntegrityError
from django.http import JsonResponse
from django.core.exceptions import ObjectDoesNotExist, ValidationError


def get_image_url(image_path):    
    """Return the URL of an image given its file path."""
    image_url = settings.MEDIA_URL + os.path.relpath(image_path, settings.MEDIA_ROOT).replace("\\", "/") 
    return image_url


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

        # Check if any images were added
        if not image_data:
            print("No images found in the Medsession directory: %s", medsession_dir)
            logger.info("No images found in the Medsession directory: %s", medsession_dir)
            
    except FileNotFoundError:
        # Handle the error if the medsession directory does not exist
        print("Medsession directory does not exist.")
    except PermissionError:
        # Handle the error if there are insufficient permissions to read the directory or a file
        print("Insufficient permissions to read the medsession directory or a file.")
    except Exception as e:
        # Handle all other exceptions
        logger.error("An error occurred while reading the medsession directory: %s", str(e))
        print("An error occurred while reading the medsession directory.")

    # Return the list of image URLs and paths
    return image_data


def get_label(person):
    """Return the corrected_label if it exists, otherwise return the elected_label."""
    if hasattr(person, 'corrected_label'):
        return person.corrected_label if person.corrected_label else person.elected_label
    elif hasattr(person, 'label'):
        return person.label
    else:
        raise AttributeError("The object doesn't have a label or corrected_label attribute.")
    
def get_person_label(person):
    """Return a label for a given Person object, in the format 'regno_Full Name'."""
    return f"{person.regno}_{person.fullname}"

def get_personal_photo_url(label):
    """Return the URL of the personal photo for a given label, or None if no such photo exists."""
    personal_photo_dir = os.path.join(settings.MEDIA_ROOT, 'y3.personal', label)
    try:
        personal_photo_files = os.listdir(personal_photo_dir)
    except FileNotFoundError:
        personal_photo_files = []

    if personal_photo_files:
        personal_photo_path = os.path.join(personal_photo_dir, personal_photo_files[0])
        # print(f'[DEBUG] Reading personal photo: {personal_photo_path}')
        personal_photo_url = get_image_url(personal_photo_path)
        # print(f'[DEBUG] Reading personal_photo_url: {personal_photo_url}')
    else:
        personal_photo_url = None

    return personal_photo_url



def get_medsessionperson_face_url(medsession, label, status, image_path, box):
    """Return the URL of the person image, processing and saving the image if necessary."""
    image_url= ''
    if status == '2' or status == '4' and box is not None:
        temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", prefix=f"{label}_", dir=temp_dir) as temp_file:
            save_path = temp_file.name

        medsession_images_dir = get_medsession_images_dir(medsession)
        if medsession_images_dir is None:
            medsession_images_dir = os.path.join(settings.STATIC_ROOT, "images")  # Default directory

        src_face_path = os.path.join(medsession_images_dir, image_path)
        
        # # If the image file doesn't exist, use the fallback image instead
        if os.path.isfile(src_face_path):
            u.save_face_from_src(src_face_path, box, save_path)
            image_url= get_image_url(save_path)

    return image_url

def medsession_persons(request, medsession_id):
    # print(f"[DEBUG] inside medsession_persons: {medsession_id}")
    logger.info("Getting medsession with sessionid %s", medsession_id)
    medsession = get_object_or_404(Medsession, sessionid=medsession_id)
    logger.debug("Found medsession: %s", medsession)
    logger.debug("Getting persons for medsession %s", medsession)

    # medsessionpersons = MedsessionPerson.objects.filter(medsession=medsession)
    medsessionpersons = MedsessionPerson.objects.select_related('person').filter(medsession=medsession)

    groups = (Person.objects
              .filter(year=medsession.year)
              .values_list('group', flat=True)
              .annotate(total=Count('group'))
              .order_by('group'))

    selected_groups = MedsessionGroup.objects.filter(medsession_id=medsession_id)
    selected_groups = [group.group for group in selected_groups if group.group is not None]
    available_groups = [group for group in groups if group not in selected_groups]
    available_groups = sorted(available_groups, key=int)
    selected_groups = sorted(selected_groups, key=int)

    logger.debug("Found medsessionpersons: %s", medsessionpersons)
    
    students_in_medsession_year = Person.objects.filter(year=medsession.year).order_by('fullname')

    # students = Person.objects.all().order_by('fullname')    

    # Get the images for the medsession
    try:
        src_image_data = get_medsession_images(medsession)
    except Exception as e:
        logger.error(f"Error getting medsession images: {str(e)}")
        src_image_data = []

    # Check if there are any images, students, or persons
    no_images = not bool(src_image_data)
    # no_students = students_in_medsession_year.count() == 0
    # no_medsessionpersons = medsessionpersons.count() == 0
    no_students = not students_in_medsession_year.exists()
    no_medsessionpersons = not medsessionpersons.exists()

    logger.info(f"src_image_data: {src_image_data}")
    
    # Iterate over students to add personal_photo_url
    for student in students_in_medsession_year:
        label = get_person_label(student)
        student.personal_photo_url = get_personal_photo_url(label)

    # Create a list of Person objects for each MedsessionPerson
    medsessionpersons_people = [mp.person for mp in medsessionpersons]
    # Filter students_in_medsession_year to only include students who are in medsessionpersons
    students_in_medsessionpersons = [student for student in students_in_medsession_year if student in medsessionpersons_people]

    students_in_medsession_year = [student for student in students_in_medsession_year if student not in students_in_medsessionpersons]
    students_in_medsession_year = sorted(
            students_in_medsession_year, 
            key=lambda student: (int(student.group), student.fullname)
        )
    
    updated_people = []
    updated_medsessionpersons = []

    for medsessionperson in medsessionpersons:
        person = medsessionperson.person
        label = get_person_label(person)

        # Check if the personal_photo_url exists, if not, get it, update and use.
        
        if not person.personal_photo_url:
            new_personal_photo_url = get_personal_photo_url(label)
            if new_personal_photo_url:  # Check if the URL obtained is not None or empty
                person.personal_photo_url = new_personal_photo_url
                updated_people.append(person)

        # Check if the image_url exists, if not, get it, update and use.
        if medsessionperson.box and not medsessionperson.image_url:
            new_image_url = get_medsessionperson_face_url(medsession, label, medsessionperson.status, medsessionperson.image_path, medsessionperson.box)
            if new_image_url:  # Check if the URL obtained is not None or empty
                medsessionperson.image_url = new_image_url
                updated_medsessionpersons.append(medsessionperson)
                logger.info("updated_medsessionpersons: (%s)", medsessionperson.image_url)

    # Bulk update all modified Person instances.
    Person.objects.bulk_update(updated_people, ['personal_photo_url'])
    MedsessionPerson.objects.bulk_update(updated_medsessionpersons, ['image_url'])
        
    # print(f"[DEBUG] medsession.sessionid = {medsession.sessionid}")
    return render(request, 'medapp/medsession_persons.html', {'src_image_data': src_image_data, 
                                                               'medsession': medsession, 
                                                               'medsessionpersons': medsessionpersons, 
                                                            #    'students': students,
                                                               'no_images': no_images,
                                                               'no_students': no_students,
                                                               'no_medsessionpersons': no_medsessionpersons,
                                                               'available_groups': available_groups, 
                                                               'selected_groups': selected_groups,
                                                               'all_groups': groups,
                                                               'students_in_medsessionpersons': students_in_medsessionpersons, 
                                                               'students_in_medsession_year': students_in_medsession_year})

def get_task_status(request, medsession_id):
    print(f"Inside get_task_status: medsession_id: {medsession_id}")
    task_status = TaskStatus.objects.filter(medsession_id=medsession_id).first()

    if task_status:
        task_status.read = True
        task_status.save()

    # If no task_status found, return a default status
    if not task_status:
        return JsonResponse({
            'status': 'no task',
            'msg': 'No task found',
            'progress': 0,
            'TimeTaken': 0,
        })
    # Prepare the response
    response = JsonResponse({
        'status': task_status.get_status_display(),
        'msg': task_status.msg,
        'progress': task_status.progress,
        'TimeTaken': task_status.TimeTaken,
    })
    # If the task is complete, delete it
    if task_status.status in ['DONE', 'FAIL']:
        task_status.delete()        
    return response

def publish_medsession_id_to_redis(medsession_id):
    print(f"Inside publish_medsession_id_to_redis: medsession_id: {medsession_id}")
    with transaction.atomic():
        # Fetch task status
        task_status = TaskStatus.objects.select_for_update().filter(medsession_id=medsession_id).first()
        # If there's an ongoing task, don't publish and report back
        if task_status and task_status.status not in ['completed', 'error']:
            return JsonResponse({
                'message': 'Task is currently running. Please wait until it is completed.',
                'status': 'error'
            })
        # If a task is 'completed' or 'error', delete it
        if task_status:
            task_status.delete()
    # Publish to Redis
    u.r.publish('medsession-channel', json.dumps({'medsession_id': medsession_id}))
    return JsonResponse({
        'message': 'Task started successfully.',
        'status': 'success'
    })

def get_medsession_images_dir(medsession):
    """Return the directory where the images for a given medsession are stored."""
    images_dir = os.path.join(
        settings.MEDIA_ROOT, 
        'runs', 
        str(medsession.year), 
        str(medsession.term), 
        str(medsession.day), 
        str(medsession.period), 
        str(medsession.room)
    )

    # Create the directory if it doesn't exist
    os.makedirs(images_dir, exist_ok=True)

    return images_dir

@csrf_exempt
@require_POST
def reprocess_images(request):            
    logger.debug(f"Inside reprocess_images: request.POST['medsession_id']: {request.POST['medsession_id']}")
    try:
        logger.debug(f"request.POST['medsession_id']: {request.POST['medsession_id']}")
        medsession_id = request.POST['medsession_id']        
        publish_medsession_id_to_redis(medsession_id)
        return JsonResponse({'message': f'Medsession {medsession_id} reprocessing started.'}, status=200)       
    except KeyError as e:
        messages.error(request, 'Medsession ID is missing in the request')
        return JsonResponse({'error': 'Medsession ID is missing in the request'}, status=400)
    except Medsession.DoesNotExist:
        messages.error(request, 'Medsession not found')
        return JsonResponse({'error': 'Medsession not found'}, status=404)
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Redis connection error: {str(e)}")
        messages.error(request, "Internal error, please try again later.")
        return JsonResponse({'error': "Internal error, please try again later."}, status=500)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)  # This will log the traceback
        messages.error(request, "Unexpected error, please try again later.")
        return JsonResponse({'error': "Unexpected error, please try again later."}, status=500)


def get_persons_in_group(year, group):
    return Person.objects.filter(year=year, group=group)


logger = logging.getLogger(__name__)

def create_medsession(request, medsession_id=None):
    medsession = None
    context = {}
    if medsession_id:
        # Editing an existing Medsession instance
        medsession = get_object_or_404(Medsession, sessionid=medsession_id)
        form = MedsessionForm(request.POST or None, instance=medsession, initial={'medsession_id': medsession_id})
        context['mode'] = 'edit'
    else:
        form = MedsessionForm(request.POST or None)
        context['mode'] = 'create'
    
    context['form'] = form    
    if form.is_valid():
        try:
            with transaction.atomic():
                medsession = form.save()
        except ValidationError as e:
            messages.error(request, str(e))
            return render(request, 'create_medsession.html', context)

        if medsession_id:  # Edit mode
            return redirect('medsession_list')
        else:  # Create mode
            return redirect('medsession_persons', medsession_id=medsession.sessionid)
    else:
        logger.error(f"Form errors: {form.errors}")
        return render(request, 'create_medsession.html', context)
    
@csrf_exempt
@require_POST
def delete_image(request, image_path):    
    try:
        # Split the image_path into medsession_id and image_name
        medsession_id, image_name = image_path.split('_', 1)        
        # Get the Medsession object
        medsession = get_object_or_404(Medsession, sessionid=medsession_id)
        print(f"[DEBUG] medsession: {medsession}")
        # Get the directory of the image
        image_dir = get_medsession_images_dir(medsession)        
        # Construct the full path of the image
        full_image_path = os.path.join(image_dir, image_name)
        # Check if the file exists
        if not os.path.isfile(full_image_path):
            return JsonResponse({'error': 'File does not exist'}, status=400)
        
         # Before deleting the file, reset MedsessionPerson related to the image and Medsession
        medsession_people = MedsessionPerson.objects.filter(medsession=medsession, image_path=image_name, status='2')  # status '2' means 'By AI'
        for person in medsession_people:
            person.status= 1
            person.attendance= 0
            person.elected_label= ''
            person.corrected_label= ''
            person.image_path = None
            person.image_url = None
            person.box = None
            person.confidence = None
            person.save()

        # Delete the file
        os.remove(full_image_path)
        return JsonResponse({'message': 'Image deleted'})
    except Medsession.DoesNotExist:
        return JsonResponse({'error': 'Medsession not found'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_POST
def upload_image(request):    
    try:
        medsession_id = request.POST['medsession_id']
        print(f"[DEBUG] medsession_id: {medsession_id}")
        medsession = get_object_or_404(Medsession, sessionid=medsession_id)
        
        # time.sleep(20)
        # Ensure an image file was uploaded
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image file uploaded'}, status=400)
        image_file = request.FILES['image']
        location=get_medsession_images_dir(medsession)
        logger.info("Medsession images directory: %s", location)
        fs = FileSystemStorage(location)
        logger.info("Medsession images FileSystemStorage: %s", fs)        
        filename = fs.save(image_file.name, image_file)
        logger.info("Medsession image saved is: %s", filename)    
        return JsonResponse({'message': 'Image uploaded'})
    except Medsession.DoesNotExist:
        return JsonResponse({'error': 'Medsession not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)    

import re
@csrf_exempt
def add_person(request, medsession_id):
    
    if request.method == "POST":
        studentLabel = request.POST.get('studentLabel')
        regno_matches = re.findall(r'\b\d+\b', studentLabel)
        regno = regno_matches[1] if len(regno_matches) > 1 else None        
        # get the medsession instance
        medsession = Medsession.objects.get(sessionid=medsession_id)        
        print(f'[DEBUG] regno: {regno}')
       
        person_to_add = get_object_or_404(Person, regno=regno)      
        print(f'[DEBUG] person_to_add: {person_to_add}')

        # Check if the corrected_person already exists in this medsession
        try:
            existing_medsession_person = MedsessionPerson.objects.get(
                medsession_id=medsession.sessionid,
                person=person_to_add
            )

            # If it exists, return a message
            return JsonResponse({"message": "Person already exists in the session", "status": "error"})

        except MedsessionPerson.DoesNotExist:
            # If it doesn't exist, create a new MedsessionPerson instance
            person = MedsessionPerson(
                medsession=medsession,
                person=person_to_add,
                added_by='1',  # for manually added
                attendance='0',  # Not Marked
                status='1',  # New Added
            )
            person.save()

            # return a success message
            return JsonResponse({"message": "Person successfully added", "status": "success"})
    else:
        # return an error message if the method is not POST
        return JsonResponse({"message": "Invalid request method", "status": "error"})
    
@csrf_exempt
@require_POST
def delete_person(request):
    person_id = request.POST.get('medsessionperson_id')
    person = get_object_or_404(MedsessionPerson, id=person_id)
    person.delete()
    return JsonResponse({
        'message': 'Person deleted successfully.'
    })

@csrf_exempt
def correct_person(request, medsessionperson_id):    
    if request.method == 'POST':
        corrected_label = request.POST.get('corrected_label')

        # Get the current MedsessionPerson, its associated Medsession and Person
        current_medsession_person = get_object_or_404(MedsessionPerson, id=medsessionperson_id)
        medsession = current_medsession_person.medsession
        current_person = current_medsession_person.person

        print(f'[DEBUG] current_person: {current_person}')
        print(f'[DEBUG] medsession: {medsession}')

        # Get the Person object for the corrected_label
        corrected_person = get_object_or_404(Person, label=corrected_label)

        # Define fields_to_copy
        fields_to_copy = ['box', 'confidence', 'image_path', 
                          'image_url', 'resnet50_label', 'resnet50_confidence', 
                          'senet50_label', 'senet50_confidence', 'vgg16_label', 
                          'vgg16_confidence', 'elected_label', 'added_by']

        # Check if the corrected_person already exists in this medsession
        try:
            existing_medsession_person = MedsessionPerson.objects.get(
                medsession_id=medsession.sessionid,
                person=corrected_person
            )

            # If it exists, overwrite it with the current MedsessionPerson data
            for field in fields_to_copy:
                setattr(existing_medsession_person, field, getattr(current_medsession_person, field))

            existing_medsession_person.corrected_label = corrected_label
            existing_medsession_person.person = corrected_person
            existing_medsession_person.attendance = '1'  # 'Present'
            existing_medsession_person.status = '4'  # 'Corrected'
            existing_medsession_person.save()

        except MedsessionPerson.DoesNotExist:
            # If it doesn't exist, move the current MedsessionPerson data to a new record
            new_medsession_person = MedsessionPerson(
                medsession=medsession,
                corrected_label=corrected_label,
                person=corrected_person,
                attendance='1',  # 'Present'
                status='4',  # 'Corrected'
                **{field: getattr(current_medsession_person, field) for field in fields_to_copy}
            )
            new_medsession_person.save()

        # Reset the old MedsessionPerson to defaults
        current_medsession_person.box = None
        current_medsession_person.confidence = None
        current_medsession_person.image_path = None
        current_medsession_person.image_url = None
        current_medsession_person.resnet50_label = None
        current_medsession_person.resnet50_confidence = None
        current_medsession_person.senet50_label = None
        current_medsession_person.senet50_confidence = None
        current_medsession_person.vgg16_label = None
        current_medsession_person.vgg16_confidence = None
        current_medsession_person.elected_label = None
        current_medsession_person.corrected_label = None
        current_medsession_person.attendance = '0'  # Reset to 'Not Marked'
        current_medsession_person.status = '1'  # Reset to 'New Added'
        current_medsession_person.added_by = '0'  # Reset to an appropriate default value
        current_medsession_person.save()

        return JsonResponse({'message': 'Changes saved successfully'})

    else:
        return HttpResponseBadRequest("Invalid request method")
    
def render_to_pdf(template_src, context_dict={}):
    # If the image_paths exist in the context
    
    # if 'image_paths' in context_dict:
        # Update each image path to the absolute path
        # context_dict['image_paths'] = [os.path.join(settings.MEDIA_URL, path) for path in context_dict['image_paths']]
    
    template = get_template(template_src)
    html = template.render(context_dict)
    result = BytesIO()   

    pdf = pisa.pisaDocument(BytesIO(html.encode("ISO-8859-1")), result, link_callback=link_callback)
    print(f"[DEBUG] inside render_to_pdf: pdf.err = {pdf.err}")
    if not pdf.err:
        with open('aaa.pdf', 'wb') as output_file:
            output_file.write(result.getvalue())
        return FileResponse(result, content_type='application/pdf')

    return None

def link_callback(uri, rel):
    # print(f"[DEBUG] inside link_callback: uri = {uri}, rel = {rel}")
    if uri.startswith(settings.MEDIA_URL):
        path = os.path.join(settings.MEDIA_ROOT, uri.replace(settings.MEDIA_URL, ""))
    elif uri.startswith(settings.STATIC_URL):
        path = os.path.join(settings.STATIC_ROOT, uri.replace(settings.STATIC_URL, ""))
    else:
        return uri
    # print(f"[DEBUG] inside link_callback: os.path.abspath(path) = {os.path.abspath(path)}")
    return os.path.abspath(path)

def download_pdf(request, medsession_id):
    # Get the same data as in your original view
    persons = MedsessionPerson.objects.filter(medsession__sessionid=medsession_id)
    medsession = get_object_or_404(Medsession, sessionid=medsession_id)
    image_paths = set()
    build_medsessions_persons(medsession, persons, image_paths)

    # ... any other data gathering and processing you need ...
    print(f"[DEBUG] inside download_pdf view: medsession.sessionid = {medsession.sessionid}")
    return render_to_pdf('medapp/medsession_persons.html', {'persons': persons, 'medsession': medsession, 'image_paths': list(image_paths)})

def create_person(request):
    if request.method == "POST":
        form = PersonForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('person_list')
    else:
        form = PersonForm()
    return render(request, 'create_person.html', {'form': form})

# ... your other views ...

class MedsessionUpdateView(UpdateView):
    model = Medsession
    template_name = 'create_medsession.html'  # Use the same template as the create view
    fields = ['year', 'term', 'day', 'room', 'course', 'lecturer'] 

    def get_success_url(self):
        return reverse_lazy('medsession_list')  # Replace 'medsession_list' with the appropriate URL name for the list view
    
class MedsessionListView(ListView):
    model = Medsession
    context_object_name = 'medsessions'
    template_name = 'medsession_list.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        medsession_filter = MedsessionFilter(self.request.GET, queryset=self.get_queryset())
        context['medsession_filter'] = medsession_filter
        return context

    def get_queryset(self):
        queryset = super().get_queryset()
        year = self.request.GET.get('year')

        if year:
            queryset = queryset.filter(year=year)
        
        return queryset

class MedsessionDeleteView(DeleteView):
    model = Medsession
    template_name = 'medsession_delete.html'  # specify the name of your template
    success_url = reverse_lazy('medsession_list')

class PersonListView(ListView):
    model = Person
    context_object_name = 'persons'
    template_name = 'person_list.html'

def manage_groups(request, medsession_sessionid):
    try:
        # Get the updated groups from the request
        updated_groups = request.POST.getlist('updated_groups[]')
        logger.info(f'Updated groups: {updated_groups}')

        # Get the current Medsession object
        medsession = Medsession.objects.get(sessionid=medsession_sessionid)

        # Get the old groups related to this Medsession
        old_groups = MedsessionGroup.objects.filter(medsession=medsession)
        logger.info(f'Old groups: {old_groups.count()}')

        # Find removed groups and delete them
        removed_groups = old_groups.exclude(group__in=updated_groups)
        removed_groups.delete()
        logger.info(f'Removed groups: {removed_groups.count()}')

        # Process new groups
        new_groups_list = [MedsessionGroup(medsession=medsession, group=group, year=medsession.year) for group in updated_groups if not MedsessionGroup.objects.filter(medsession=medsession, group=group).exists()]
        MedsessionGroup.objects.bulk_create(new_groups_list)
        logger.info(f'Created new MedsessionGroups: {len(new_groups_list)}')

        # Add related persons to MedsessionPerson
        for group in updated_groups:
            group_persons = Person.objects.filter(group=group)
            new_persons_list = [MedsessionPerson(medsession=medsession, person=person, added_by='0', status='1', attendance='0') for person in group_persons if not MedsessionPerson.objects.filter(medsession=medsession, person=person).exists()]
            MedsessionPerson.objects.bulk_create(new_persons_list)
            logger.info(f'Created new MedsessionPersons: {len(new_persons_list)}')

        # After updating the groups, remove any person that is not in any MedsessionGroup and its status is not manual
        persons_to_remove = MedsessionPerson.objects.filter(medsession=medsession).exclude(Q(person__group__in=updated_groups) | Q(status='3'))
        persons_to_remove.delete()
        logger.info(f'Removed persons: {persons_to_remove.count()}')

        return JsonResponse({'message': 'Changes saved successfully'})
    except Medsession.DoesNotExist:
        logger.error('Medsession does not exist')
        return JsonResponse({'message': 'Medsession does not exist'}, status=400)
    except ValidationError as e:
        logger.error(f'Validation error: {e}')
        return JsonResponse({'message': f'Validation error: {e}'}, status=400)
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        return JsonResponse({'message': f'Unexpected error: {e}'}, status=500)
    
@require_POST
def mark_medsessionperson(request, medsessionperson_id):
    medsessionperson = get_object_or_404(MedsessionPerson, id=medsessionperson_id)

    attendance = request.POST.get('attendance')
    status = request.POST.get('status')
    added_by = request.POST.get('added_by')

    if attendance in dict(MedsessionPerson.ATTENDANCE_CHOICES):
        if attendance == '0':
            status = '1'
            added_by = '0'
        else:
            if status == '1' or status == '2' or status == '5':
                status = '3'
            added_by = '1'

        if status in dict(MedsessionPerson.STATUS_CHOICES) and added_by in dict(MedsessionPerson.ADDED_CHOICES):
            medsessionperson.attendance = attendance
            medsessionperson.status = status
            medsessionperson.added_by = added_by
            medsessionperson.save()
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'success': False, 'error': 'Invalid input.'})
    else:
        return JsonResponse({'success': False, 'error': 'Invalid input.'})
    
import numpy as np
def export_medsessionpersons(request, sessionid):
    try:
        # Fetch the medsession
        medsession = Medsession.objects.get(sessionid=sessionid)

        # Fetch the associated medsession persons
        medsessionpersons = MedsessionPerson.objects.filter(medsession=medsession)
        
        # Prepare data for the Excel file
        data = []
        for medsessionperson in medsessionpersons:
            label = medsessionperson.corrected_label if medsessionperson.corrected_label else medsessionperson.elected_label

            row = {
                'Group': medsessionperson.person.group,  # Include group                
                'Registration': medsessionperson.person.regno,
                'Full Name': medsessionperson.person.fullname,
                # 'Label': label,
                'Attendance': dict(MedsessionPerson.ATTENDANCE_CHOICES)[medsessionperson.attendance],  # Include attendance
                # 'Detection Status': dict(MedsessionPerson.STATUS_CHOICES)[medsessionperson.status],  # Include detection status
                # 'Added By': 'AI' if medsessionperson.added_by == '0' else 'Manual',      
            }
            data.append(row)

        # Create a pandas DataFrame
        df = pd.DataFrame(data)

        # Convert 'Group' column to integers for sorting
        df['Group'] = df['Group'].apply(lambda x: int(x) if x.isdigit() else np.inf)
        # Sort the DataFrame by 'Group', 'Full Name', 'Attendance', 'Detection Status', and 'Added By'
        df = df.sort_values(by=['Group', 'Full Name', 'Attendance'])

        # Create a BytesIO buffer
        buffer = io.BytesIO()

        # Create an Excel writer with this buffer as file
        excel_writer = pd.ExcelWriter(buffer, engine="xlsxwriter")

        # Convert the dataframe to an XlsxWriter Excel object
        df.to_excel(excel_writer, index=False, sheet_name='Report')

        # Get the xlsxwriter workbook and worksheet objects
        workbook  = excel_writer.book
        worksheet = excel_writer.sheets['Report']

        # Set the column width based on the max length in each column
        for column in df:
            column_length = max(df[column].astype(str).map(len).max(), len(column))
            col_idx = df.columns.get_loc(column)
            worksheet.set_column(col_idx, col_idx, column_length + 4)  # added a small buffer for better aesthetics 

        # Close the Excel writer and output the Excel file to the buffer
        excel_writer.close()

        # # Set the column width
        # # 'A:D' means columns from A to D, and 20 is the width
        # worksheet.set_column('A:D', 30)
        # # Close the Excel writer and output the Excel file to the buffer
        # excel_writer.close()
    except Exception as e:
        # Handle other exceptions such as issues with DataFrame, ExcelWriter, etc.
        raise e

    # Create a HTTP response
    response = HttpResponse(buffer.getvalue(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = f'attachment; filename=Year_{medsession.year}_Term_{medsession.term}_Day_{medsession.day}_Period_{medsession.period}_Room_{medsession.room}.xlsx'

    return response


def attendance_list_calendar(request):
    return render(request, 'attendance_list_calendar.html')


import datetime

def medsession_data_calendar(request):
    medsessions = Medsession.objects.all()  # Or filter as needed

    # Define your period times
    period_times = {
        1: ('09:00:00', '10:30:00'),
        2: ('11:00:00', '12:30:00'),
        3: ('13:00:00', '14:30:00'),
        # Add more if you have more periods
    }

    medsession_list = []
    for medsession in medsessions:
        # Construct the title
        title = f"Year {medsession.year} - Term {medsession.term} - Room {medsession.room} - {medsession.course if medsession.course else ''}"
        
        # Get the start and end times for this medsession's period
        start_time, end_time = period_times[int(medsession.period)]
        
        # Construct the start and end datetime strings
        start = f"{medsession.day}T{start_time}"  # assuming day is a date string in format 'YYYY-MM-DD'
        end = f"{medsession.day}T{end_time}"  # assuming day is a date string in format 'YYYY-MM-DD'
        
        medsession_list.append({
            'id': medsession.sessionid,
            'title': title,
            'start': start,
            'end': end,
            # Add other properties as needed
        })

    return JsonResponse(medsession_list, safe=False)