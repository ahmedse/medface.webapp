
# Create your views here.
import os
from django.shortcuts import render, redirect, get_object_or_404
from .forms import MedsessionForm, PersonForm, ImageForm
from .models import Medsession, Person, MedsessionPerson, TaskStatus
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
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Person
import pandas as pd
import io
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.http import Http404
from django.db import transaction
import traceback
from django.shortcuts import redirect


def get_image_url(image_path):    
    """Return the URL of an image given its file path."""
    image_url = settings.MEDIA_URL + os.path.relpath(image_path, settings.MEDIA_ROOT).replace("\\", "/") 
    return image_url

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
    if hasattr(person, 'corrected_label'):
        return person.corrected_label if person.corrected_label else person.elected_label
    elif hasattr(person, 'label'):
        return person.label
    else:
        raise AttributeError("The object doesn't have a label or corrected_label attribute.")

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

import os

def get_person_image_url(medsession, label, image_path, box):
    """Return the URL of the person image, processing and saving the image if necessary."""
    temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", prefix=f"{label}_", dir=temp_dir) as temp_file:
        save_path = temp_file.name

    medsession_images_dir = get_medsession_images_dir(medsession)
    if medsession_images_dir is None:
        medsession_images_dir = os.path.join(settings.STATIC_ROOT, "images")  # Default directory

    if image_path is None:
        image_path = "no-preview.png"  # Default image

    src_face_path = os.path.join(medsession_images_dir, image_path)

    # If the image file doesn't exist, use the fallback image instead
    if not os.path.isfile(src_face_path):
        src_face_path = os.path.join(settings.STATIC_ROOT, "images/no-preview.png")

    u.save_face_from_src(src_face_path, box, save_path)

    return get_image_url(save_path)


def medsession_persons(request, medsession_id):
    # print(f"[DEBUG] inside medsession_persons: {medsession_id}")
    logger.debug("Getting medsession with sessionid %s", medsession_id)
    medsession = get_object_or_404(Medsession, sessionid=medsession_id)
    logger.debug("Found medsession: %s", medsession)
    logger.debug("Getting persons for medsession %s", medsession)
    persons = MedsessionPerson.objects.filter(medsession=medsession)
    logger.debug("Found persons: %s", persons)
    students = Person.objects.all().order_by('fullname')

    # Get the images for the medsession
    src_image_data = get_medsession_images(medsession)
    logger.info(f"src_image_data: {src_image_data}")
    # print(f"[DEBUG] src_image_data: {src_image_data}")

        # Iterate over students to add personal_photo_url
    for student in students:
        label = get_label(student)
        student.personal_photo_url = get_personal_photo_url(label)

    for person in persons:
        label = get_label(person)
        registration, name = label.split('_')
        person.registration = registration
        person.name = name
        person.personal_photo_url = get_personal_photo_url(label)

        if not person.image_url:
            person.image_url = get_person_image_url(medsession, label, person.image_path, person.box)

        person.added_by = '1' if person.corrected_label else '0'
        person.save()
        
        # print(f"[DEBUG] medsession.sessionid = {medsession.sessionid}")
    return render(request, 'medapp/medsession_persons.html', {'src_image_data': src_image_data, 'medsession': medsession, 'persons': persons, 'students': students})

import redis
import json
from django.http import JsonResponse


def get_task_status(request, medsession_id):
    print(f"Inside get_task_status: medsession_id: {medsession_id}")
    task_status = TaskStatus.objects.filter(medsession_id=medsession_id).first()
    # If no task_status found, return a default status
    if not task_status:
        return JsonResponse({
            'status': 'no task',
            'progress': 0,
            'TimeTaken': 0,
        })
    # Prepare the response
    response = JsonResponse({
        'status': task_status.status,
        'progress': task_status.progress,
        'TimeTaken': task_status.TimeTaken,
    })
    # If the task is complete, delete it
    if task_status.status in ['completed', 'error']:
        task_status.delete()        
    return response

from django.db import transaction

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


def create_medsession(request):
    ImageFormSet = modelformset_factory(Image, form=ImageForm, extra=6, max_num=10, min_num=1)

    if request.method == "POST":
        return handle_create_medsession_post(request, ImageFormSet)
    else:
        return handle_create_medsession_get(request, ImageFormSet)

def handle_create_medsession_post(request, ImageFormSet):
    form = MedsessionForm(request.POST)
    formset = ImageFormSet(request.POST, request.FILES, queryset=Image.objects.none())

    context = {'form': form, 'formset': formset}
    if form.is_valid() and formset.is_valid():
        medsession = form.save()
        for image_form in formset:
            if image_form.cleaned_data:
                image = image_form.save(commit=False)
                image.medsession = medsession
                image.save()
        try:
            publish_medsession_id_to_redis(medsession.sessionid)
        except ValidationError as e:
            messages.error(request, str(e))
            return render(request, 'create_medsession.html', context)
        return redirect('medsession_persons', medsession_id=medsession.sessionid)
    else:
        logger.error(f"Form errors: {form.errors}")
        return render(request, 'create_medsession.html', context)

def handle_create_medsession_get(request, ImageFormSet):
    form = MedsessionForm()
    formset = ImageFormSet(queryset=Image.objects.none())
    field_names = ['year', 'term', 'day', 'room', 'course', 'lecturer']
    context = {'form': form, 'formset': formset, 'field_names': field_names}
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
        # Delete the file
        os.remove(full_image_path)
        return JsonResponse({'message': 'Image deleted'})
    except Medsession.DoesNotExist:
        return JsonResponse({'error': 'Medsession not found'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

import time
from django.core.files.storage import FileSystemStorage
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
        fs = FileSystemStorage(location=get_medsession_images_dir(medsession))
        filename = fs.save(image_file.name, image_file)
        return JsonResponse({'message': 'Image uploaded'})
    except Medsession.DoesNotExist:
        return JsonResponse({'error': 'Medsession not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)    

def export_medsessionpersons(request, sessionid):
    # Fetch the medsession
    medsession = Medsession.objects.get(sessionid=sessionid)

    # Fetch the associated medsession persons
    medsessionpersons = MedsessionPerson.objects.filter(medsession=medsession)

    # Prepare data for the Excel file
    data = []
    for person in medsessionpersons:
        label = person.corrected_label if person.corrected_label else person.elected_label
        registration, fullname = label.split('_')
        person.registration = registration
        person.fullname = fullname
        row = {
            'Registration': person.registration,
            'Full Name': person.fullname,
            'Label': label,
            'Added By': 'AI' if person.added_by == '0' else 'Manual',
        }
        data.append(row)

    # Create a pandas DataFrame
    df = pd.DataFrame(data)

    # Create a BytesIO buffer
    buffer = io.BytesIO()

    # Create an Excel writer with this buffer as file
    excel_writer = pd.ExcelWriter(buffer, engine="xlsxwriter")

    # Convert the dataframe to an XlsxWriter Excel object
    df.to_excel(excel_writer, index=False, sheet_name='Report')

    # Get the xlsxwriter workbook and worksheet objects
    workbook  = excel_writer.book
    worksheet = excel_writer.sheets['Report']

    # Set the column width
    # 'A:D' means columns from A to D, and 20 is the width
    worksheet.set_column('A:D', 30)
    # Close the Excel writer and output the Excel file to the buffer
    excel_writer.close()

    # Create a HTTP response
    response = HttpResponse(buffer.getvalue(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = f'attachment; filename=Year_{medsession.year}_Term_{medsession.term}_Day_{medsession.day}_Period_{medsession.period}_Room_{medsession.room}.xlsx'

    return response

@csrf_exempt
def add_person(request):
    if request.method == "POST":
        corrected_label = request.POST.get('corrected_label')
        medsession_id = request.POST.get('medsession_id')

        # get the medsession instance
        medsession = Medsession.objects.get(sessionid=medsession_id)

        # create a new MedsessionPerson instance
        person = MedsessionPerson.objects.create(
            medsession=medsession,
            corrected_label=corrected_label,
            added_by='1'  # for manually added
        )
        # return a success message
        return JsonResponse({"message": "Person successfully added", "status": "success"})
    else:
        # return an error message if the method is not POST
        return JsonResponse({"message": "Invalid request method", "status": "error"})

@csrf_exempt
@require_POST
def delete_person(request):
    person_id = request.POST.get('person_id')
    person = get_object_or_404(MedsessionPerson, id=person_id)
    person.delete()
    return JsonResponse({
        'message': 'Person deleted successfully.'
    })

@csrf_exempt
def correct_person(request, person_id):    
    if request.method == 'POST':
        corrected_label = request.POST.get('corrected_label')
        person = MedsessionPerson.objects.get(id=person_id)
        person.corrected_label = corrected_label
        person.save()
        print(f"[DEBUG] correct label: {person.elected_label} to  {person.corrected_label}  ")
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

def create_medsession_OLD(request):        
    # ImageFormSet = modelformset_factory(Image, fields=('image',), extra=3, max_num=10)
    ImageFormSet = modelformset_factory(Image, form=ImageForm, extra=6, max_num=10, min_num=1)
    if request.method == "POST":
        form = MedsessionForm(request.POST)
        formset = ImageFormSet(request.POST, request.FILES, queryset=Image.objects.none())
        # print(request.POST)  # Print the POST data
        context = {'form': form, 'formset': formset} 
        if form.is_valid() and formset.is_valid():
            medsession = form.save()
            image_paths = []
            for image_form in formset:
                if image_form.cleaned_data:
                    image = image_form.save(commit=False)
                    image.medsession = medsession
                    image.save()
                    image_paths.append(image.image.path)                
            try:    
                # Now, perform face recognition
                publish_medsession_id_to_redis(medsession.sessionid)
            except ValidationError as e:
                messages.error(request, str(e))
                return render(request, 'create_medsession.html', context)
            # 6- redirect to the medsession_person view details page. 
            return redirect('medsession_persons', medsession_id=medsession_persons[0].medsession_id)
            # return redirect('medsession_list')
        else:
            print(form.errors)
    else:
        form = MedsessionForm()
        formset = ImageFormSet(queryset=Image.objects.none())        
    field_names = ['year', 'term', 'day', 'room', 'course', 'lecturer']
    context = {'form': form, 'formset': formset, 'field_names': field_names}
    return render(request, 'create_medsession.html', context)

def create_person(request):
    if request.method == "POST":
        form = PersonForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('person_list')
    else:
        form = PersonForm()
    return render(request, 'create_person.html', {'form': form})

def build_medsessions_persons(medsession, persons, image_paths):
    for person in persons:
        registration, name = person.elected_label.split('_')
        person.registration = registration
        person.name = name
        
        image_path = os.path.join(settings.MEDIA_ROOT, 'runs', str(medsession.year)
                            , str(medsession.term), str(medsession.day)
                            , str(medsession.period), str(medsession.room)
                            , str(person.image_path))
            
        image_url = settings.MEDIA_URL + os.path.relpath(image_path, settings.MEDIA_ROOT).replace("\\", "/")
        image_paths.add(image_url)

        # Check if the person already has an image_url
        if not person.image_url:
            # Generate a unique temporary save path
            temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")   
            # Create the temp directory if it does not exist
            os.makedirs(temp_dir, exist_ok=True)        
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=temp_dir)
            save_path = temp_file.name
            temp_file.close()

            # Process the image
            u.process_image(image_path, person.box, save_path)  # assuming person.box has the bounding box coordinates

            # Generate the image URL
            image_url = settings.MEDIA_URL + os.path.relpath(save_path, settings.MEDIA_ROOT).replace("\\", "/")
            person.image_url = image_url


# ... your other views ...

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

class MedsessionUpdateView(UpdateView):
    model = Medsession
    template_name = 'medsession_edit.html'  # specify the name of your template
    fields = ['year', 'term', 'day', 'room', 'course', 'lecturer']  # specify the fields to be edited

    def get_success_url(self):
        return reverse_lazy('medsession_list')

class MedsessionDeleteView(DeleteView):
    model = Medsession
    template_name = 'medsession_delete.html'  # specify the name of your template
    success_url = reverse_lazy('medsession_list')

class PersonListView(ListView):
    model = Person
    context_object_name = 'persons'
    template_name = 'person_list.html'