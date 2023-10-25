
# Create your views here.
import os
from django.shortcuts import render, redirect, get_object_or_404
from .forms import MedsessionForm, PersonForm, ImageForm
from .models import Medsession, Person, MedsessionPerson
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
logger = logging.getLogger(__name__)
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Person
import pandas as pd
import io

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

def create_medsession(request):        
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
                # 0- some settings                
                medsession_dir = os.path.join(settings.MEDIA_ROOT, 'runs'
                                              , str(medsession.year), str(medsession.term)
                                              , str(medsession.day), str(medsession.period), str(medsession.room))
                # 1- loop on images and extract faces and store in faces_df
                faces_df = u.extract_faces(medsession_dir)
                print(f'[DEBUG] {len(faces_df)}')
                # 2- query the 3 models and store in relevant column in person_df
                person_df = u.query_models(faces_df)
                # 3- elect one answer by: the max agreed one, or the max whieghted by confidence.
                elected = u.elect_answer(person_df)
                # 4- remove duplicated persons
                unique_persons = u.remove_duplicates(elected)
                # 5- create medsession_person object and store in db.
                medsession_persons = u.create_medsession_persons(unique_persons, medsession)
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


def medsession_persons(request, medsession_id):
    logger.debug("Getting medsession with sessionid %s", medsession_id)
    medsession = get_object_or_404(Medsession, sessionid=medsession_id)
    logger.debug("Found medsession: %s", medsession)
    logger.debug("Getting persons for medsession %s", medsession)
    persons = MedsessionPerson.objects.filter(medsession=medsession)
    logger.debug("Found persons: %s", persons)
    students = Person.objects.all().order_by('fullname')

    # Split the elected_label into registration and name for each person
    image_paths = set()
    persons_with_photos = []
    for person in persons:
        # Use corrected_label if it exists, otherwise use elected_label
        label = person.corrected_label if person.corrected_label else person.elected_label
        registration, name = label.split('_')
        person.registration = registration
        person.name = name
        
        # Create the path to the directory where the person's personal photos are stored
        personal_photo_dir = os.path.join(settings.MEDIA_ROOT, 'y3.personal', '{}'.format(label))
        try:
            personal_photo_files = os.listdir(personal_photo_dir)
        except FileNotFoundError:
            personal_photo_files = []
            
        if personal_photo_files:
            personal_photo_path = os.path.join(personal_photo_dir, personal_photo_files[0])
            personal_photo_url = settings.MEDIA_URL + os.path.relpath(
                personal_photo_path, 
                settings.MEDIA_ROOT
            ).replace("\\", "/")
        else:
            personal_photo_url = None
        person.personal_photo_url = personal_photo_url
        
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
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", prefix=f"{label}_", dir=temp_dir)
            save_path = temp_file.name
            temp_file.close()

            # Process the image
            u.process_image(image_path, person.box, save_path)  # assuming person.box has the bounding box coordinates

            # Generate the image URL
            image_url = settings.MEDIA_URL + os.path.relpath(save_path, settings.MEDIA_ROOT).replace("\\", "/")
            person.image_url = image_url
        
        # Set added_by field based on whether the corrected_label exists or not
        person.added_by = '1' if person.corrected_label else '0'

        person.save()
        
        # print(f"[DEBUG] medsession.sessionid = {medsession.sessionid}")

    return render(request, 'medapp/medsession_persons.html', {'persons': persons, 'medsession': medsession, 'image_paths': list(image_paths), 'students': students})

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