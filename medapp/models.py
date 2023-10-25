from django.db import models
# Create your models here.
from django.utils import timezone
from django.db import models

class Medsession(models.Model):
    PERIOD_CHOICES = [(str(i), str(i)) for i in range(1, 4)]
    YEAR_CHOICES = [(str(i), str(i)) for i in range(1, 7)]
    TERM_CHOICES = [(str(i), str(i)) for i in range(1, 4)]

    sessionid = models.AutoField(primary_key=True)
    year = models.CharField(max_length=1, choices=YEAR_CHOICES, default='3')
    term = models.CharField(max_length=1, choices=TERM_CHOICES, default='1')
    day = models.DateField(default=timezone.now)
    period = models.CharField(max_length=1, choices=PERIOD_CHOICES, default='1')   
    room = models.CharField(max_length=50)
    course = models.CharField(max_length=400, default='n/a')
    lecturer = models.CharField(max_length=400, default='n/a')
    modifytime = models.DateTimeField(auto_now=True)
 
def upload_to(instance, filename):
    return f'runs/{instance.medsession.year}/{instance.medsession.term}/{instance.medsession.day}/{instance.medsession.period}/{instance.medsession.room}/{filename}'
 
class Image(models.Model):
    medsession = models.ForeignKey(Medsession, related_name='images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to=upload_to)
 
class MedsessionPerson(models.Model):
    ADDED_CHOICES = [(str(i), str(i)) for i in range(0, 1)]
    
    medsession = models.ForeignKey(Medsession, on_delete=models.CASCADE)    
    box = models.TextField( null=True)  # assuming box is a list that can be represented as a string
    confidence = models.FloatField( null=True)
    image_path = models.TextField( null=True)
    image_url = models.CharField(max_length=200,  null=True)
    resnet50_label = models.CharField(max_length=200,  null=True)
    resnet50_confidence = models.FloatField(null=True)
    senet50_label = models.CharField(max_length=200,  null=True)
    senet50_confidence = models.FloatField( null=True)
    vgg16_label = models.CharField(max_length=200,  null=True)
    vgg16_confidence = models.FloatField( null=True)
    elected_label = models.CharField(max_length=200)
    corrected_label = models.CharField(max_length=200,  null=True)
    added_by= models.CharField(max_length=1, choices=ADDED_CHOICES, default='0')

class Person(models.Model):
    YEAR_CHOICES = [(str(i), str(i)) for i in range(1, 7)]
    GROUP_CHOICES = [(str(i), str(i)) for i in range(1, 12)]

    id = models.AutoField(primary_key=True)
    regno = models.CharField(max_length=15, unique=True)
    fullname = models.CharField(max_length=450)
    arname = models.CharField(max_length=450, default='')
    year = models.CharField(max_length=1, choices=YEAR_CHOICES, default='3')
    year_b = models.CharField(max_length=20, default='22-26')
    group = models.CharField(max_length=2, choices=GROUP_CHOICES)
    label = models.CharField(max_length=100, default='')
    
    modifytime = models.DateTimeField(auto_now=True)
    