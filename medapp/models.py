from django.db import models
# Create your models here.
from django.utils import timezone
from django.db import models
from django.db.models import Avg

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

    @property
    def detected_faces_count(self):
        return self.medsessionperson_set.filter(status='2').count()

    @property
    def manual_entries_count(self):
        return self.medsessionperson_set.filter(status='3').count()

    @property
    def corrections_count(self):
        return self.medsessionperson_set.filter(status='4').count()

    @property
    def average_confidence(self):
        return self.medsessionperson_set.aggregate(Avg('confidence'))['confidence__avg']
 
class TaskStatus(models.Model):
    STATUS_CHOICES = [
        ('INIT', 'Initializing'),
        ('PROG', 'In Progress'),
        ('DONE', 'Done'),
        ('FAIL', 'Failed'),
    ]
    medsession = models.ForeignKey(Medsession, on_delete=models.CASCADE)
    status = models.CharField(max_length=255, choices=STATUS_CHOICES)
    msg= models.CharField(max_length=255)
    progress = models.IntegerField(default=0)
    TimeTaken = models.IntegerField(default=0)
    read = models.BooleanField(default=False)
    last_updated = models.DateTimeField(auto_now=True)

def upload_to(instance, filename):
    return f'runs/{instance.medsession.year}/{instance.medsession.term}/{instance.medsession.day}/{instance.medsession.period}/{instance.medsession.room}/{filename}'
 
 
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
    personal_photo_url = models.URLField(blank=True, null=True)  # new field
    modifytime = models.DateTimeField(auto_now=True)

class Image(models.Model):
    medsession = models.ForeignKey(Medsession, related_name='images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to=upload_to)
 
class MedsessionPerson(models.Model):
    ADDED_CHOICES = [
        ('0', 'By AI'),
        ('1', 'Manual'),
    ]
    YEAR_CHOICES = [(str(i), str(i)) for i in range(1, 7+1)]
    GROUP_CHOICES = [(str(i), str(i)) for i in range(1, 12+1)]       
    ATTENDANCE_CHOICES = [
        ('0', 'Not Marked'),
        ('1', 'Present'),
        ('2', 'Absent'),
        ('3', 'Excused Absence'),
        ('4', 'Tardy/Late'),
        ('5', 'Early Departure'),
        ('6', 'Partial Attendance'),
        ('7', 'Present with Permission')
    ]
    STATUS_CHOICES = [
        ('1', 'New Added'),
        ('2', 'By AI'),
        ('3', 'Manual'),
        ('4', 'Corrected'),
        ('5', 'Not Detected'),
        ('6', 'Unknown ')
    ]
    
    medsession = models.ForeignKey(Medsession, on_delete=models.CASCADE) 
    person = models.ForeignKey(Person, on_delete=models.CASCADE)
    attendance= models.CharField(
        max_length=200,
        choices=ATTENDANCE_CHOICES,
        default='0'
    )
    status = models.CharField(
        max_length=200,
        choices=STATUS_CHOICES,
        default='1'
    )
    id = models.AutoField(primary_key=True)
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
    elected_label = models.CharField(max_length=200, null=True)
    corrected_label = models.CharField(max_length=200,  null=True)
    added_by = models.CharField(max_length=1, choices=ADDED_CHOICES, default='0')

    class Meta:
        unique_together = (('medsession', 'person'),)

class MedsessionGroup(models.Model):
    YEAR_CHOICES = [(str(i), str(i)) for i in range(1, 7+1)]
    GROUP_CHOICES = [(str(i), str(i)) for i in range(1, 12+1)]    
    medsession = models.ForeignKey(Medsession, on_delete=models.CASCADE)    
    year = models.TextField( null=True)  # assuming box is a list that can be represented as a string
    group = models.IntegerField( null=True)
    modifytime = models.DateTimeField(auto_now=True)


class Room(models.Model):
    BUILDING_CHOICES = [('B2', 'B2'), ('B3', 'B3'), ('B4', 'B4'), ('B5', 'B5'), ('B6', 'B6'), ('Hospital', 'Hospital'), ('Other', 'Other')]
    FLOOR_CHOICES = [(str(i), str(i)) for i in range(0, 5)]
    SECTION_CHOICES = [('PBL', 'PBL'), ('Clinical', 'Clinical'), ('Bedside Teaching', 'Bedside Teaching')]
    TYPE_CHOICES = [('Lecture Room', 'Lecture Room'), ('Skill Lab', 'Skill Lab'), ('Think Tank', 'Think Tank'), ('Virtual Hospital', 'Virtual Hospital'), ('X-Reality', 'X-Reality'), ('Amphitheatre', 'Amphitheatre'), ('Operating Theatre', 'Operating Theatre')]

    id = models.AutoField(primary_key=True)
    room_code = models.CharField(max_length=15, unique=False, blank=True)
    room_section = models.CharField(max_length=50, choices=SECTION_CHOICES, default='', null=True, blank=True)
    notes = models.CharField(max_length=250, default='', null=True, blank=True)
    type = models.CharField(max_length=50, choices=TYPE_CHOICES, default='Lecture Room', null=True, blank=True)
    floor = models.CharField(max_length=50, choices=FLOOR_CHOICES, default='', null=True, blank=True)
    building = models.CharField(max_length=50, choices=BUILDING_CHOICES, default='B2', null=True, blank=True)
    modifytime = models.DateTimeField(auto_now=True)



    