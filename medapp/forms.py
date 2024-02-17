from django import forms
from .models import Medsession, Person, Image, Room, MedsessionGroup
import datetime
from django.forms import modelformset_factory

class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ('image', )

    def clean_image(self):
        image = self.cleaned_data.get('image', False)
        if image:
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError('Image file too large ( > 10mb )')
        return image

ImageFormSet = modelformset_factory(Image, form=ImageForm, extra=3, max_num=10)

class RoomModelChoiceField(forms.ModelChoiceField):
    def label_from_instance(self, obj):
        return f"{obj.building}- Floor: {obj.floor} - {obj.room_code} [{obj.type}]"
    
class MedsessionForm(forms.ModelForm):
    # ImageFormSet = modelformset_factory(Image, fields=('image',), extra=3, max_num=10)    
    PERIOD_CHOICES = [
        ('1', '9-11 AM'),
        ('2', '11-1 PM'),
        ('3', '1-3 PM'),
    ]

    period = forms.ChoiceField(choices=PERIOD_CHOICES, widget=forms.RadioSelect)
    day = forms.DateField(
        initial=datetime.date.today().strftime('%Y-%m-%d'), 
        widget=forms.DateInput(attrs={'type': 'date', 'value': datetime.date.today().strftime('%Y-%m-%d')})
    )

    room = RoomModelChoiceField(
        queryset=Room.objects.all().order_by('building', 'floor', 'room_code'),
        empty_label=None,
    )

    class Meta:
        model = Medsession
        fields = ['year', 'period', 'day', 'term', 'room', 'course', 'lecturer']

    def label_from_instance(self, obj):
        return f"{obj.building}-{obj.floor}-{obj.room_code} [{obj.type}]"
    
    def clean_room(self):
        room = self.cleaned_data.get('room')
        if room:
            return room.room_code
        return room

class PersonForm(forms.ModelForm):
    class Meta:
        model = Person
        fields = ['regno', 'fullname', 'year']


class GroupForm(forms.ModelForm):
    groups = forms.ModelMultipleChoiceField(queryset=MedsessionGroup.objects.all(), 
                                            widget=forms.CheckboxSelectMultiple)
    class Meta:
        model = MedsessionGroup
        fields = ['group']