from django.urls import path
from .views import create_medsession, create_person, PersonListView, MedsessionListView, MedsessionUpdateView, MedsessionDeleteView
from . import views 
from django.conf import settings
from django.conf.urls.static import static
from .views import reprocess_images, upload_image, delete_image, medsession_persons, download_pdf, correct_person, add_person
from .views import get_task_status, delete_person, manage_groups, mark_medsessionperson

urlpatterns = [
    path('medsession_list/', MedsessionListView.as_view(), name='medsession_list'),
  
    path('create_medsession/', views.create_medsession, name='create_medsession'),
    path('create_medsession/<int:medsession_id>/', views.create_medsession, name='edit_medsession'),

    path('medsessions/delete/<int:pk>/', MedsessionDeleteView.as_view(), name='medsession_delete'),
    
    path('medsession/<int:medsession_id>/persons/', views.medsession_persons, name='medsession_persons'),
    path('medsession/<int:medsession_id>/pdf', download_pdf, name='medsession_pdf'),
    path('correct_person/<int:medsessionperson_id>/', views.correct_person, name='correct_person'),

    path('add_person/<int:medsession_id>/', views.add_person, name='add_person'),
    path('export_medsessionpersons/<int:sessionid>/', views.export_medsessionpersons, name='export_medsessionpersons'),
    path('delete_image/<str:image_path>/', views.delete_image, name='delete_image'),
    path('upload_image/', views.upload_image, name='upload_image'),
    path('reprocess_images/', views.reprocess_images, name='reprocess_images'),
    path('get_task_status/<int:medsession_id>/', get_task_status, name='get_task_status'),
    path('delete_person/', delete_person, name='delete_person'),
    # path('get_task_status/<int:medsession_id>/', get_task_status, name='get_task_status'),
    path('medsession/<int:medsession_sessionid>/manage-groups/', manage_groups, name='manage-groups'),
    path('mark_medsessionperson/<int:medsessionperson_id>/', mark_medsessionperson, name='mark_medsessionperson'),
   
       
    path('create_person/', create_person, name='create_person'),
    path('person_list/', PersonListView.as_view(), name='person_list'),
]

# for debug purposes
def print_urls(urllist, indent=0):
    for entry in urllist:
        print("  " * indent, entry.pattern)
        if hasattr(entry, 'url_patterns'):
            print_urls(entry.url_patterns, indent + 1)

# print_urls(urlpatterns)